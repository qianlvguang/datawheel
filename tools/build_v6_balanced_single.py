#!/usr/bin/env python3
"""Datawheel v2 → Qwen3.5-27B 中训练 V6 通道「balanced 单图行」构建器。

把 v2 **train split only** 的 stage_b / stage_c / stage_e 单图记录扩写成
ms-swift 4.3.2 中训练 parquet 行：

    columns = messages / images / enable_thinking / sample_id / cate_id / variant

所有行 `variant=6`（训练插件自动映射 channel='V6' → loss_V6），
`enable_thinking=False`，`images` 只引用 v2 已有相对路径（不复制/不生成图片）。

精确 cate 配额（合计 37,186 行，逐行精确）：

    apparel 6,453 / flag-hard 6,453 / ip-brand 3,131 / anime-ip 3,131 /
    misleading 4,791 / quantity 3,833 / text-compliance 3,914 /
    structured_caption 5,480

行类型：
  1) multi_turn_qa      —— apparel / flag-hard / ip-brand / anime-ip / misleading /
                           quantity，每行 2~5 轮（misleading 2~4 轮）user/assistant QA；
  2) text_compliance    —— 判断描述与图是否一致，约一半一致/一半故意篡改（1~2 轮）；
  3) structured_caption —— 严格 JSON 结构化描述（1 轮，2 种指令模板）。

多视图策略（配额的主要来源）：
  - 同一张 v2 图可生成多条**不同任务视图**的行；同一 (cate, 图片) 内的每一行至少
    第一问不同（确定性 anchor 机制），因此同图多行内容不会完全相同；
  - 每行 `sample_id` 唯一（`V6B_S_{cate}_{序号:06d}`），行级键
    `(sample_id, cate_id, images)` 唯一；
  - 每个 cate 独立生成到配额，图片顺序固定按 `v2_sample_id` 排序、视图数量按
    确定性的 Bresenham 均摊；配额无法达到时显式报错（`QuotaError`），绝不静默凑数。

数据卫生：
  - 只取 `split=="train"` 的 v2 记录，绝不使用 val / test_unseen_*（manifest 记录
    `split_policy="train_only"`）；
  - 所有新增问答的事实（名称/bbox/颜色/位置/计数/方位/工艺/品类）都来自 v2 记录字段或
    由 entities_json/bboxes 直接计算，不编造；
  - structured_caption 的 assistant 输出为 `json.dumps(..., ensure_ascii=False)` 合法 JSON；
  - 文本长度门禁：中文 chars/1.6 + 其余 chars/4 + 图片×512 ≤ 8192，超长时修剪轮数。

用法（在 Coding/datawheel/ 下）：
  .venv/bin/python tools/build_v6_balanced_single.py
  .venv/bin/python tools/build_v6_balanced_single.py --seed 42
  .venv/bin/python tools/validate_swift_v6.py \
      --dataset output/datawheel_v6_balanced/parts/single_rows.parquet \
      --image-root . --out output/datawheel_v6_balanced/parts/validation_single/

产物：
  output/datawheel_v6_balanced/parts/single_rows.parquet
  output/datawheel_v6_balanced/parts/single_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "output" / "datawheel_v2" / "dataset.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_balanced" / "parts"
OUT_PARQUET = "single_rows.parquet"
OUT_MANIFEST = "single_manifest.json"

VARIANT = 6
SEED = 42
MAX_LEN = 8192
IMAGE_TOKEN_ESTIMATE = 512

STAGE_B = "stage_b_single_element"
STAGE_C = "stage_c_multi_element"
STAGE_E = "stage_e_negative_evidence"
STAGE_BCE = (STAGE_B, STAGE_C, STAGE_E)

# ---------------------------------------------------------------------------
# 任务配额 / 多视图上限
# ---------------------------------------------------------------------------
CATE_TARGETS: "OrderedDict[str, int]" = OrderedDict([
    ("apparel", 6453),
    ("flag-hard", 6453),
    ("ip-brand", 3131),
    ("anime-ip", 3131),
    ("misleading", 4791),
    ("quantity", 3833),
    ("text-compliance", 3914),
    ("structured_caption", 5480),
])
CATE_ORDER = tuple(CATE_TARGETS)
TOTAL_ROWS = sum(CATE_TARGETS.values())
assert TOTAL_ROWS == 37186, TOTAL_ROWS

MULTI_TURN_CATES = ("apparel", "flag-hard", "ip-brand", "anime-ip", "misleading", "quantity")
MAX_VIEWS = {
    "apparel": 2,
    "flag-hard": 3,
    "ip-brand": 2,
    "anime-ip": 2,
    "misleading": 6,
    "quantity": 2,
    "structured_caption": 2,
}

# ---------------------------------------------------------------------------
# 领域 / 词表常量
# ---------------------------------------------------------------------------
OFFICIAL_DOMAINS = ("official_flags", "official_emblems", "official_regional")
BRAND_DOMAINS = ("luxury_brands", "ip_brands")
ANIME_DOMAINS = ("character_ips", "subculture_elements")
CATE_DOMAIN_GROUPS = {
    "flag-hard": OFFICIAL_DOMAINS,
    "ip-brand": BRAND_DOMAINS,
    "anime-ip": ANIME_DOMAINS,
}

DOMAIN_LABEL_ZH = {
    "official_flags": "国家/地区主权旗帜",
    "official_emblems": "官方国徽/纹章",
    "official_regional": "地方行政区旗帜",
    "luxury_brands": "高奢品牌徽标",
    "ip_brands": "商业/娱乐 IP 品牌徽标",
    "character_ips": "动漫/游戏角色立绘",
    "subculture_elements": "亚文化潮流图元",
}

CATEGORY_IDENTIFY_Q = {
    "flag-hard": "图中的官方旗帜/徽记图案是什么？",
    "ip-brand": "图中的品牌徽标图案是什么？",
    "anime-ip": "图中的角色/IP 图案是什么？",
}
CATEGORY_SURFACE_Q = {
    "flag-hard": "图中商品表面是否有官方旗帜/徽记印花？",
    "ip-brand": "图中商品表面是否有品牌徽标印花？",
    "anime-ip": "图中商品表面是否有动漫角色或亚文化图元印花？",
}

APPAREL_ZH_KEYWORDS = ("上装", "裙装", "连衣裙", "裤装", "下装", "箱包", "鞋履", "帽饰",
                       "大衣", "风衣", "衬衫", "夹克", "毛衣")
APPAREL_EN_KEYWORDS = (
    "top", "t-shirt", "tshirt", "tee", "dress", "pants", "trousers", "jeans",
    "bag", "backpack", "tote", "shoe", "sneaker", "hat", "headwear", "cap",
    "beanie", "shirt", "coat", "jacket", "sweater", "cardigan", "bottomwear",
    "outerwear", "knitwear", "blouse",
)

# 合规判定「篡改工艺位置」时使用的跨品类区位词（确保与真值语义不同）
ZONE_GROUP_LABELS = {
    "bag": ["包身正面主视觉印制区", "包盖/正面外袋装饰徽标区"],
    "hat": ["帽冠正面主视觉区"],
    "shoe": ["鞋面/鞋帮主标识区", "鞋舌/鞋面小标区"],
    "top": ["前胸中央主视觉印花区", "左胸上方小型徽标区", "下摆边缘织唛/小标区"],
    "bottom": ["裤腿正面/侧面印花区", "下摆边缘织唛/小标区"],
    "skirt": ["裙摆边缘下摆小标区"],
}
ZONE_ID_GROUP = {
    "bag_front_panel": "bag",
    "bag_front_flap": "bag",
    "hat_front_panel": "hat",
    "shoe_lateral_quarter": "shoe",
    "shoe_tongue_panel": "shoe",
    "center_chest_graphic": "top",
    "bodice_center_graphic": "top",
    "left_chest_badge": "top",
    "lower_hem_patch": "top",
    "pants_hem_patch": "bottom",
    "thigh_side_graphic": "bottom",
    "skirt_hem_patch": "skirt",
}
ZONE_FALLBACK_LABEL = {
    "bag": "包身正面主视觉印制区",
    "hat": "帽冠正面主视觉区",
    "shoe": "鞋面/鞋帮主标识区",
    "top": "前胸中央主视觉印花区",
    "bottom": "裤腿正面/侧面印花区",
    "skirt": "裙摆边缘下摆小标区",
}

RELATION_FAKE = ("上方", "上方左侧", "上方右侧", "正左侧", "正右侧")

IDENTIFY_Q = (
    "图中的印花图案是什么？",
    "请识别商品上印制/贴合的主要视觉图元。",
    "这件商品上的图案内容是什么？",
)

CAPTION_PROMPTS = (
    "请以严格 JSON 输出该商品的结构化描述：品类、工艺位置、印花元素（名称/边界框/颜色）、"
    "面料与场景。只输出 JSON，不要输出其他内容。",
    "请观察图片并输出严格 JSON（不要 Markdown、不要解释），字段为 category_zh、"
    "scene_style_zh、placement_zone、print_technique_zh、elements（含 name_zh、bbox_1000、"
    "color_zh）与 caption_zh。",
)

COMPLIANCE_PROMPTS = (
    "判断以下描述与图片是否一致，并给出简短依据：{statement}",
    "请核对下面的描述与图像是否相符，并简要说明依据：{statement}",
    "以下关于图片的文字描述是否正确？请判断并简述依据。描述：{statement}",
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")
DISPLAY_RE = re.compile(r"【([^】]+)】")
OCCLUDED_ZONE_RE = re.compile(r"无法确认：(.+?)被同色面料")

MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("messages", MESSAGES_SCHEMA),
    ("images", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
    ("sample_id", pa.string()),
    ("cate_id", pa.string()),
    ("variant", pa.int32()),
])

# 运行期全局词表（由 load_source 填充，内容只来自 v2 train）
DOMAIN_VOCAB: Dict[str, List[Dict[str, str]]] = {}
TITLE_COLOR: Dict[str, str] = {}

COLOR_VOCAB: List[str] = []
TECH_VOCAB: List[str] = []
POSITION_VOCAB: List[str] = []


class QuotaError(RuntimeError):
    """cate 配额在可用图片池上无法达到时抛出（不静默凑数）。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> float:
    """中文 chars/1.6 + 其余 chars/4（与 tools/validate_swift_v6.py 公式一致）。"""
    if not text:
        return 0.0
    cjk = len(CJK_RE.findall(text))
    return cjk / 1.6 + (len(text) - cjk) / 4.0


def messages_text(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def row_token_estimate(messages: Sequence[Dict[str, str]], n_images: int) -> float:
    return estimate_text_tokens(messages_text(messages)) + n_images * IMAGE_TOKEN_ESTIMATE


def trim_messages_to_limit(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """超长时从尾部成对修剪轮数；单轮仍超长则抛错（不静默截断）。"""
    while len(messages) > 2 and row_token_estimate(messages, 1) > MAX_LEN:
        messages = messages[:-2]
    if row_token_estimate(messages, 1) > MAX_LEN:
        raise ValueError(f"单轮文本估算 {row_token_estimate(messages, 1):.1f} > {MAX_LEN}")
    return messages


def fmt_bbox(bbox: Sequence[int]) -> str:
    return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"


def bbox_area(bbox: Sequence[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def rng_for(seed: int, *parts: Any) -> random.Random:
    """由 seed + 字符串路径派生确定性随机流（与迭代顺序无关）。"""
    return random.Random("|".join([f"seed={seed}"] + [str(p) for p in parts]))


def plan_views(target: int, pool_size: int, max_views: int, cate: str) -> List[int]:
    """把 target 行按确定性 Bresenham 均摊到 pool_size 张图上，返回每张图的视图数。

    配额不足（target > pool_size * max_views）时抛出 QuotaError，并给出可达到的最大行数。
    """
    if pool_size <= 0:
        raise QuotaError(f"cate={cate} 候选池为 0，无法生成 {target} 行")
    max_rows = pool_size * max_views
    if target > max_rows:
        raise QuotaError(
            f"cate={cate} 配额不足：需要 {target} 行，但 pool={pool_size} 张图 × "
            f"max_views={max_views} 最多只能生成 {max_rows} 行（缺口 {target - max_rows}）")
    base, rem = divmod(target, pool_size)
    if base < 1:
        raise QuotaError(f"cate={cate} target={target} < pool={pool_size}，请改用 select_evenly")
    return [base + (((i + 1) * rem) // pool_size - (i * rem) // pool_size)
            for i in range(pool_size)]


def select_evenly(pool: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """从 pool 中确定性地均匀抽取 n 个元素（保持原顺序，n ≤ len(pool)）。"""
    size = len(pool)
    if n > size:
        raise QuotaError(f"select_evenly 需要 {n} > pool={size}")
    if n == size:
        return list(pool)
    idxs = [(i * size) // n for i in range(n)]
    # 防御：矩形除法在 n<=size 时严格递增，此处只需断言
    if len(set(idxs)) != n:
        raise QuotaError("select_evenly 索引重复（不应发生）")
    return [pool[i] for i in idxs]


def dedupe_items(items: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen, out = set(), []
    for question, answer in items:
        q = str(question).strip()
        a = str(answer).strip()
        if not q or not a or q in seen:
            continue
        seen.add(q)
        out.append((q, a))
    return out


def choose_rounds(cate: str, rng: random.Random, n_items: int) -> int:
    if cate == "misleading":
        rounds = rng.choices([2, 3, 4], weights=[3, 4, 3], k=1)[0]
    elif cate == "quantity":
        rounds = rng.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    else:
        rounds = rng.choices([2, 3, 4, 5], weights=[2, 3, 3, 2], k=1)[0]
    return max(2, min(rounds, n_items))


def question_priority(question: str) -> int:
    """问答条目的自然顺序权重（越小越适合作为开场/靠前）。"""
    if any(key in question for key in ("颜色", "色调", "代表色", "底色")):
        return 4  # 颜色
    if ("边界框" in question) or ("定位" in question):
        return 3  # 定位
    if any(key in question for key in ("几处", "几个主要", "一共", "数量")):
        return 1  # 计数
    if any(key in question for key in ("是否存在", "是否印有", "能看到", "是否出现",
                                       "能否确认", "请判断", "请确认", "是否为",
                                       "是否相同", "可观察", "可见", "是否有证据",
                                       "能否根据", "是否有", "局部或整体")):
        return 2  # 存在性 / 证据判断
    if any(key in question for key in ("位置", "哪里", "方位", "位于", "哪个区域", "方向")):
        return 5  # 位置 / 方位
    if ("是什么" in question) or ("识别" in question) or ("图案内容" in question):
        return 0  # 识别
    return 6


def select_round_items(
    items: Sequence[Tuple[str, str]],
    rounds: int,
    rng: random.Random,
    view: int,
    used_anchors: Sequence[str] = (),
    required: Sequence[Tuple[str, str]] = (),
) -> List[Tuple[str, str]]:
    """选择 rounds 条问答：开场问题按 view 轮换（保证同图多视图首问不同），
    必须包含 required 中的关键问答（如 quantity 的计数、misleading 的否定/证据不足），
    其余问题避免与已选问答同答案，并按自然提问顺序排列。"""
    n = len(items)
    if n < 2:
        raise ValueError("可用问答条目不足 2 条")
    rounds = max(2, min(rounds, n))
    opening_indices = [i for i in range(n) if question_priority(items[i][0]) <= 2]
    used = set(used_anchors)
    anchor = None
    if opening_indices:
        for offset in range(len(opening_indices)):
            candidate = opening_indices[(view + offset) % len(opening_indices)]
            if items[candidate][0] not in used:
                anchor = candidate
                break
    if anchor is None:
        for offset in range(n):
            candidate = (view + offset) % n
            if items[candidate][0] not in used:
                anchor = candidate
                break
    if anchor is None:
        anchor = view % n

    question_index: Dict[str, int] = {}
    for index, (question, _answer) in enumerate(items):
        question_index.setdefault(question, index)

    chosen: List[int] = [anchor]
    chosen_questions = {items[anchor][0]}
    chosen_answers = {items[anchor][1]}
    for req_question, req_answer in required:
        if len(chosen) >= rounds:
            break
        req_index = question_index.get(req_question)
        if req_index is None or req_index in chosen:
            continue
        if req_answer in chosen_answers:
            continue
        chosen.append(req_index)
        chosen_questions.add(req_question)
        chosen_answers.add(req_answer)

    rest = [i for i in range(n) if i not in chosen]
    rng.shuffle(rest)  # attempt 变化时可选集合不同（配合 commit_row 去重）
    for index in rest:
        if len(chosen) >= rounds:
            break
        if items[index][1] in chosen_answers:
            continue
        chosen.append(index)
        chosen_answers.add(items[index][1])
    if len(chosen) < rounds:
        for index in rest:
            if len(chosen) >= rounds:
                break
            if index in chosen:
                continue
            chosen.append(index)
    tail = chosen[1:]
    tail.sort(key=lambda index: question_priority(items[index][0]))
    chosen = [chosen[0]] + tail
    return [items[i] for i in chosen]


def shift_bbox(bbox: Sequence[int], rng: random.Random) -> List[int]:
    ymin, xmin, ymax, xmax = (int(v) for v in bbox)
    for _ in range(32):
        dy = rng.choice([-120, -95, -80, 80, 95, 120])
        dx = rng.choice([-110, -85, 0, 85, 110])
        if dy == 0 and dx == 0:
            continue
        shifted = [
            max(0, min(1000, ymin + dy)),
            max(0, min(1000, xmin + dx)),
            max(0, min(1000, ymax + dy)),
            max(0, min(1000, xmax + dx)),
        ]
        if shifted != [ymin, xmin, ymax, xmax] and shifted[0] < shifted[2] and shifted[1] < shifted[3]:
            return shifted
    raise ValueError(f"无法构造与真值不同的 bbox：{list(bbox)}")


def pick_other(values: Sequence[str], true_value: str, rng: random.Random) -> Optional[str]:
    pool = [v for v in values if v and v != true_value]
    return rng.choice(pool) if pool else None


def is_apparel_substrate(row: Dict[str, Any]) -> bool:
    zh = (row.get("substrate_category_zh") or "").lower()
    en = (row.get("substrate_category_en") or "").lower()
    return any(k in zh for k in APPAREL_ZH_KEYWORDS) or any(k in en for k in APPAREL_EN_KEYWORDS)


def canonical_zone_group(rec: Dict[str, Any]) -> Optional[str]:
    return ZONE_ID_GROUP.get(rec.get("placement_zone") or "")


def pick_other_position(rec: Dict[str, Any], rng: random.Random) -> Optional[Tuple[str, str]]:
    """为不一致描述构造一个跨品类、肯定错误的工艺位置；返回 (fake, true)。"""
    true_loc = rec.get("location_answer")
    group = canonical_zone_group(rec)
    if not true_loc or not group:
        return None
    other_groups = [g for g in sorted(ZONE_GROUP_LABELS) if g != group]
    rng.shuffle(other_groups)
    for other in other_groups:
        labels = [label for label in ZONE_GROUP_LABELS[other] if label != true_loc]
        if labels:
            return rng.choice(labels), true_loc
    return None


def display_name(entity: Dict[str, Any]) -> str:
    return entity.get("full_title_zh") or entity.get("name_zh") or ""


# ---------------------------------------------------------------------------
# 读取 v2 train + 候选记录
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = [
    "sample_id", "task_stage", "task_family", "split", "domain",
    "substrate_category_zh", "substrate_category_en", "scene_style_zh",
    "placement_zone", "print_technique_zh", "bboxes_norm_1000",
    "entities_json", "entity_ids_json", "caption_zh", "qa_pairs_json",
    "image_path", "match_label", "num_entities",
]


def build_domain_vocab(train_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """domain → 该域 train 可见实体名（name_zh/stem 去重，确定性排序）。"""
    raw: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = defaultdict(dict)
    for row in train_rows:
        try:
            eids = json.loads(row.get("entity_ids_json") or "[]")
        except json.JSONDecodeError:
            continue
        for eid in eids:
            domain = eid.get("domain") or ""
            name_zh = eid.get("name_zh") or ""
            stem = eid.get("stem") or ""
            if not domain or not name_zh or not stem:
                continue
            raw[domain][(name_zh, stem)] = {
                "domain": domain,
                "name_zh": name_zh,
                "name_en": eid.get("name_en") or "",
                "stem": stem,
            }
    return {domain: [values[key] for key in sorted(values)]
            for domain, values in sorted(raw.items())}


def build_color_map(raw_records: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], str], int]:
    """(domain, stem) → 实体代表色，仅用 v2 train 的 b/c 颜色 QA 汇总。"""
    color_map: Dict[Tuple[str, str], str] = {}
    conflicts = 0
    for raw in raw_records:
        stage = raw["stage"]
        if stage not in ("b", "c"):
            continue
        colors = [q["answer_zh"] for q in raw["qa"] if q.get("answer_type") == "color"]
        if not colors:
            continue
        eids = raw["entity_ids"]
        if len(eids) == 1:
            target_idx = 0
        elif stage == "c" and len(eids) == 2:
            # stage_c 的 color QA 问的是「次要图案」，对应第 2 个实体
            target_idx = 1
        else:
            continue
        eid = eids[target_idx]
        key = (eid["domain"], eid["stem"])
        color = colors[0]
        if key in color_map:
            if color_map[key] != color:
                conflicts += 1
        else:
            color_map[key] = color
    return color_map, conflicts


def normalize_record(raw: Dict[str, Any],
                     color_map: Dict[Tuple[str, str], str]) -> Optional[Dict[str, Any]]:
    """把 v2 原始行规范化为内部记录；事实不自洽时返回 None（由调用方计数）。"""
    stage = raw["stage"]
    qa = raw["qa"]
    entities_raw = raw["entities"]
    entity_ids = raw["entity_ids"]
    bboxes = raw["bboxes"]
    if stage in ("b", "c"):
        if not entities_raw or len(entities_raw) != len(entity_ids):
            return None
        entities: List[Dict[str, Any]] = []
        for idx, (entity, eid) in enumerate(zip(entities_raw, entity_ids)):
            bbox = entity.get("bbox_1000") or (bboxes[idx] if idx < len(bboxes) else None)
            if not bbox or len(bbox) != 4:
                return None
            ymin, xmin, ymax, xmax = (int(v) for v in bbox)
            if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                return None
            color = color_map.get((eid["domain"], eid["stem"]))
            entities.append({
                "name_zh": entity.get("entity_name_zh") or eid.get("name_zh") or "",
                "name_en": entity.get("entity_name_en") or eid.get("name_en") or "",
                "full_title_zh": entity.get("full_title_zh") or eid.get("name_zh") or "",
                "bbox": [ymin, xmin, ymax, xmax],
                "color": color or "",
                "domain": eid["domain"],
                "stem": eid["stem"],
            })
    else:  # stage_e：absent/occluded 无实体；hard_distractor 只有 1 个干扰图元实体
        entities = []
        for entity in entities_raw:
            bbox = entity.get("bbox_1000")
            if not bbox or len(bbox) != 4:
                continue
            ymin, xmin, ymax, xmax = (int(v) for v in bbox)
            if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                continue
            entities.append({
                "name_zh": entity.get("entity_name_zh") or "",
                "name_en": entity.get("entity_name_en") or "",
                "full_title_zh": entity.get("full_title_zh") or entity.get("entity_name_zh") or "",
                "bbox": [ymin, xmin, ymax, xmax],
                "color": "",
                "domain": "",
                "stem": "",
            })
        if not entity_ids:
            return None

    colors = [q["answer_zh"] for q in qa if q.get("answer_type") == "color"]
    name_answers = [q["answer_zh"] for q in qa if q.get("answer_type") == "name"]
    shape_answer = None
    for q in qa:
        if q.get("answer_type") == "name" and any(k in (q.get("question_zh") or "")
                                                  for k in ("构型", "比例", "外形")):
            shape_answer = q["answer_zh"]
            break
    location_answer = next((q["answer_zh"] for q in qa if q.get("answer_type") == "location"), None)
    relation_answer = next((q["answer_zh"] for q in qa if q.get("answer_type") == "relation"), None)
    count_answer = next((q["answer_zh"] for q in qa if q.get("answer_type") == "count"), None)

    if stage in ("b", "c"):
        # 校验名称 QA 与 entities_json 主图元一致（v2 事实自洽）
        if not name_answers or entities[0]["full_title_zh"] not in name_answers:
            return None
        # 校验 b 记录颜色 QA 与颜色表一致
        if len(entities) == 1 and colors and entities[0]["color"] != colors[0]:
            return None
        if stage == "c" and len(entities) == 2 and colors and entities[1]["color"] != colors[0]:
            return None
        if any(not entity["color"] for entity in entities):
            return None
        fam = raw["family"]
    else:
        fam = raw["family"].replace("negative_", "", 1)

    rec: Dict[str, Any] = {
        "v2_sample_id": raw["v2_sample_id"],
        "stage": stage,
        "family": raw["family"],
        "domain": raw["domain"],
        "domains": sorted({eid["domain"] for eid in entity_ids}),
        "category_zh": raw["category_zh"],
        "category_en": raw["category_en"],
        "scene_style_zh": raw["scene_style_zh"],
        "placement_zone": raw["placement_zone"],
        "technique": raw["technique"],
        "caption_zh": raw["caption_zh"],
        "image_path": raw["image_path"],
        "qa": qa,
        "entities": entities,
        "entity_ids": entity_ids,
        "count": len(entities),
        "match_label": raw["match_label"],
        "name_answer": name_answers[0] if name_answers else "",
        "shape_answer": shape_answer,
        "color_answer": colors[0] if colors else None,
        "location_answer": location_answer,
        "relation_answer": relation_answer,
        "count_answer": count_answer,
        "is_apparel": raw["is_apparel"],
        "e_case": fam,
        "e_target_name_zh": entity_ids[0].get("name_zh") or "",
        "e_target_display": "",
        "e_zone_zh": "",
        "e_actual": None,
    }
    if stage == "e":
        display = None
        for q in qa:
            match = DISPLAY_RE.search(q.get("question_zh") or "")
            if match:
                display = match.group(1)
                break
        rec["e_target_display"] = display or rec["e_target_name_zh"]
        if fam == "occluded":
            zone = None
            for q in qa:
                match = OCCLUDED_ZONE_RE.search(q.get("answer_zh") or "")
                if match:
                    zone = match.group(1)
                    break
            if not zone:
                group = canonical_zone_group(rec)
                zone = ZONE_FALLBACK_LABEL.get(group or "", "目标印花区")
            rec["e_zone_zh"] = zone
        if fam == "hard_distractor" and entities:
            rec["e_actual"] = entities[0]
            short = None
            for q in qa:
                if q.get("answer_type") == "name":
                    match = DISPLAY_RE.search(q.get("answer_zh") or "")
                    if match:
                        short = match.group(1)
                        break
            rec["e_actual"]["display_short"] = short or rec["e_actual"]["name_zh"]
    return rec


def load_source(dataset_path: Path, stats: Dict[str, Any]) -> Dict[str, Any]:
    table = pq.read_table(dataset_path, columns=SOURCE_COLUMNS)
    rows = table.to_pylist()
    stats["source_rows"] = len(rows)
    stats["source_split_dist"] = dict(sorted(Counter(r["split"] for r in rows).items()))
    train_rows = [r for r in rows if r["split"] == "train"]
    stats["source_train_rows"] = len(train_rows)
    stats["source_train_stage_dist"] = dict(sorted(Counter(r["task_stage"] for r in train_rows).items()))

    domain_vocab = build_domain_vocab(train_rows)
    DOMAIN_VOCAB.clear()
    DOMAIN_VOCAB.update(domain_vocab)

    skipped: Counter = Counter()
    raw_records: List[Dict[str, Any]] = []
    for row in train_rows:
        stage = row["task_stage"]
        if stage not in STAGE_BCE:
            skipped["stage_not_bce"] += 1
            continue
        image_rel = row["image_path"] or ""
        if not image_rel or not (PROJECT_ROOT / image_rel).is_file():
            skipped["image_missing"] += 1
            continue
        try:
            qa = json.loads(row["qa_pairs_json"] or "[]")
            entities = json.loads(row["entities_json"] or "[]")
            entity_ids = json.loads(row["entity_ids_json"] or "[]")
            bboxes = json.loads(row["bboxes_norm_1000"] or "[]")
        except json.JSONDecodeError:
            skipped["json_broken"] += 1
            continue
        if not qa:
            skipped["empty_qa"] += 1
            continue
        if any(not (q.get("question_zh") or "").strip() for q in qa):
            skipped["empty_question"] += 1
            continue
        if any(not (q.get("answer_zh") or "").strip() for q in qa):
            skipped["empty_answer"] += 1
            continue
        raw_records.append({
            "v2_sample_id": row["sample_id"],
            "stage": {STAGE_B: "b", STAGE_C: "c", STAGE_E: "e"}[stage],
            "task_stage": stage,
            "family": row["task_family"] or "",
            "domain": row["domain"] or "",
            "category_zh": row["substrate_category_zh"] or "",
            "category_en": row["substrate_category_en"] or "",
            "scene_style_zh": row["scene_style_zh"] or "",
            "placement_zone": row["placement_zone"] or "",
            "technique": row["print_technique_zh"] or "",
            "caption_zh": row["caption_zh"] or "",
            "image_path": image_rel,
            "qa": qa,
            "entities": entities,
            "entity_ids": entity_ids,
            "bboxes": bboxes,
            "match_label": row["match_label"] or "",
            "is_apparel": is_apparel_substrate(row),
        })

    color_map, color_conflicts = build_color_map(raw_records)
    records: List[Dict[str, Any]] = []
    normalize_skipped: Counter = Counter()
    for raw in raw_records:
        rec = normalize_record(raw, color_map)
        if rec is None:
            normalize_skipped[f"stage_{raw['stage']}"] += 1
            continue
        records.append(rec)

    # 颜色/颜色统计词表（仅 train 汇总）
    global COLOR_VOCAB, TECH_VOCAB, POSITION_VOCAB, TITLE_COLOR
    colors, techniques, positions = set(), set(), set()
    title_color: Dict[str, str] = {}
    for rec in records:
        if rec["stage"] not in ("b", "c"):
            continue
        techniques.add(rec["technique"])
        if rec["color_answer"]:
            colors.add(rec["color_answer"])
        if rec["location_answer"]:
            positions.add(rec["location_answer"])
        for entity in rec["entities"]:
            if entity["color"]:
                colors.add(entity["color"])
                if entity["full_title_zh"]:
                    title_color.setdefault(entity["full_title_zh"], entity["color"])
    COLOR_VOCAB = sorted(colors)
    TECH_VOCAB = sorted(t for t in techniques if t)
    POSITION_VOCAB = sorted(positions)
    TITLE_COLOR.clear()
    TITLE_COLOR.update(title_color)

    # e hard_distractor 的实际图案补上颜色（来自同 title 的 b/c 实体调色板记录）
    for rec in records:
        if rec["stage"] == "e" and rec.get("e_actual"):
            rec["e_actual"]["color"] = TITLE_COLOR.get(rec["e_actual"]["full_title_zh"], "")

    stats["skipped"] = dict(sorted(skipped.items()))
    stats["normalize_skipped"] = dict(sorted(normalize_skipped.items()))
    stats["color_map_entities"] = len(color_map)
    stats["color_conflicts"] = color_conflicts
    stats["color_vocab_size"] = len(COLOR_VOCAB)
    stats["technique_vocab_size"] = len(TECH_VOCAB)
    stats["position_vocab_size"] = len(POSITION_VOCAB)
    stats["title_color_coverage"] = len(TITLE_COLOR)
    stats["records"] = len(records)
    stats["records_by_stage"] = dict(sorted(Counter(r["stage"] for r in records).items()))
    return {"records": records, "source_rows": len(rows), "train_rows": len(train_rows)}


# ---------------------------------------------------------------------------
# 问答条目（事实全部来自 v2 记录）
# ---------------------------------------------------------------------------
def pick_absent_entities(rec: Dict[str, Any],
                         rng: random.Random,
                         k: int = 2,
                         prefer_domains: Optional[Sequence[str]] = None) -> List[str]:
    """同域内未出现在本图中的 train 实体名（答案恒为「否」）。"""
    row_names = set()
    for entity in rec.get("entities") or []:
        for key in ("name_zh", "full_title_zh"):
            if entity.get(key):
                row_names.add(entity[key])
    for key in ("e_target_name_zh", "e_target_display"):
        if rec.get(key):
            row_names.add(rec[key])
    actual = rec.get("e_actual")
    if actual:
        for key in ("name_zh", "full_title_zh", "display_short"):
            if actual.get(key):
                row_names.add(actual[key])

    domains = list(rec.get("domains") or [])
    if prefer_domains:
        ordered = ([d for d in sorted(domains) if any(g in d for g in prefer_domains)]
                   + [d for d in sorted(domains) if not any(g in d for g in prefer_domains)])
    else:
        ordered = sorted(domains)
    if not ordered:
        ordered = sorted(DOMAIN_VOCAB)

    def conflicts(name: str) -> bool:
        return any(name in row_name or row_name in name for row_name in row_names if row_name)

    candidates: List[str] = []
    seen: set = set()
    for domain in ordered:
        for item in DOMAIN_VOCAB.get(domain, []):
            name = item["name_zh"]
            if not name or name in seen or conflicts(name):
                continue
            seen.add(name)
            candidates.append(name)
    if len(candidates) < k:
        for domain in sorted(DOMAIN_VOCAB):
            for item in DOMAIN_VOCAB[domain]:
                name = item["name_zh"]
                if not name or name in seen or conflicts(name):
                    continue
                seen.add(name)
                candidates.append(name)
    if not candidates:
        return []
    k = min(k, len(candidates))
    indices = sorted(rng.sample(range(len(candidates)), k))
    return [candidates[i] for i in indices]


def grounding_items(rec: Dict[str, Any], cate: str, rng: random.Random) -> List[Tuple[str, str]]:
    """stage_b / stage_c 单图 grounding+属性问答（含多视图用的新问法）。"""
    items: List[Tuple[str, str]] = [(q["question_zh"], q["answer_zh"]) for q in rec["qa"]]
    entities = rec["entities"]
    cat = rec["category_zh"]

    cate_group = CATE_DOMAIN_GROUPS.get(cate)
    domain_specific_ok = False
    domain_specific_answer = ""
    if cate_group:
        matching = [e for e in entities if any(g in e["domain"] for g in cate_group)]
        if len(matching) == 1:
            domain_specific_ok = True
            domain_specific_answer = display_name(matching[0])

    for idx, entity in enumerate(entities):
        name = display_name(entity)
        bbox = fmt_bbox(entity["bbox"])
        color = entity["color"]
        role = "主视觉图案" if idx == 0 else "次要图案"
        if idx == 0:
            for question in IDENTIFY_Q:
                items.append((question, name))
            if domain_specific_ok:
                items.append((CATEGORY_IDENTIFY_Q[cate], domain_specific_answer))
        else:
            items.append((f"{role}是什么？", name))
            items.append((f"图中的次要图案「{entity['name_zh']}」是什么？", name))
        for question in (
            f"请给出{role}的归一化边界框 [ymin,xmin,ymax,xmax]。",
            f"定位{role}，输出 [ymin,xmin,ymax,xmax] 相对坐标。",
            f"{role}在整图中的边界框坐标是多少？",
        ):
            items.append((question, bbox))
        for question in (f"{role}的主导颜色是什么？", f"请描述{role}的主色调。"):
            items.append((question, color))

    location = rec.get("location_answer")
    if location:
        items.append(("该图案位于商品的哪个工艺位置？", location))
        items.append(("印花在商品上的具体位置是？", location))

    count = rec["count"]
    count_disp = rec.get("count_answer") or f"共有 {count} 处独立印花/视觉元素。"
    items.append(("图中共有几处独立印花？", count_disp))
    items.append(("商品上共有几个主要视觉图元？", count_disp))

    relation = rec.get("relation_answer")
    if relation:
        items.append(("主视觉图案与次要图案的相对位置关系是什么？", relation))
        items.append(("次要图案相对主视觉图案位于哪个方位？", relation))
        if "右" in relation:
            items.append(("次要图案是否位于主视觉图案的右侧？", "是。"))
        if "左" in relation:
            items.append(("次要图案是否位于主视觉图案的左侧？", "是。"))
        if "下" in relation:
            items.append(("次要图案是否位于主视觉图案的下方？", "是。"))

    shape = rec.get("shape_answer")
    if shape:
        items.append(("该图案的几何构型属于哪一类？", shape))
        items.append(("该图案的整体外形比例如何？", shape))

    items.append(("图中商品属于什么品类？", cat))

    for absent in pick_absent_entities(rec, rng, k=2, prefer_domains=cate_group):
        items.append((f"图中商品上是否存在【{absent}】图案？", f"否，图中不存在【{absent}】。"))
        items.append((f"商品表面能看到【{absent}】的印花吗？", f"否，图中不存在【{absent}】。"))

    if len(entities) >= 2:
        first, second = entities[0], entities[1]
        area_first, area_second = bbox_area(first["bbox"]), bbox_area(second["bbox"])
        if area_first > area_second:
            items.append(("主视觉图案与次要图案哪一个面积更大？",
                          f"主视觉图案【{display_name(first)}】面积更大。"))
        elif area_second > area_first:
            items.append(("主视觉图案与次要图案哪一个面积更大？",
                          f"次要图案【{display_name(second)}】面积更大。"))
        domain_counts = Counter(e["domain"] for e in entities)
        for domain in sorted(domain_counts):
            label = DOMAIN_LABEL_ZH.get(domain)
            if label:
                items.append((f"图中属于「{label}」类的印花图元有几处？",
                              f"{domain_counts[domain]} 处。"))
        color_counts = Counter(e["color"] for e in entities)
        for color in sorted(color_counts):
            items.append((f"图中代表色为「{color}」的印花图元有几处？",
                          f"{color_counts[color]} 处。"))
        same_color = "相同。" if first["color"] == second["color"] else "不同。"
        items.append(("主视觉图案与次要图案的代表色是否相同？", same_color))
    return dedupe_items(items)


def negative_items(rec: Dict[str, Any], cate: str, rng: random.Random) -> List[Tuple[str, str]]:
    """stage_e 负样本/证据不足记录的可变问答条目。"""
    items: List[Tuple[str, str]] = [(q["question_zh"], q["answer_zh"]) for q in rec["qa"]]
    case = rec["e_case"]
    target = rec["e_target_display"]
    cat = rec["category_zh"]
    cate_group = CATE_DOMAIN_GROUPS.get(cate)

    if case == "absent_clean":
        color = rec.get("color_answer") or ""
        items.extend([
            (f"该商品上是否印有【{target}】图案？", f"否，图中商品表面没有【{target}】图案。"),
            (f"请确认这件商品上是否存在【{target}】的印花？",
             f"不存在：画面中没有【{target}】的任何印花像素。"),
            (f"图中是否出现了【{target}】的局部或整体？",
             f"没有，画面中未出现【{target}】的任何部分。"),
            ("图中商品表面是否存在任何印花图案？", "否，该商品表面为纯色面料，未见任何印花图案。"),
            (f"图中是否存在与【{target}】相似的印花图案？",
             "没有，画面中不存在任何印花图案，也不存在相似图案。"),
            ("商品底布是什么颜色？", color),
            ("画面中商品的面料底色是？", color),
            ("图中商品属于什么品类？", cat),
        ])
        if cate_group:
            items.append((CATEGORY_SURFACE_Q[cate], "否，该商品表面为纯色面料，未见任何印花图案。"))
        for absent in pick_absent_entities(rec, rng, k=2, prefer_domains=cate_group):
            items.append((f"图中商品上是否存在【{absent}】图案？", f"否，图中不存在【{absent}】。"))
            items.append((f"商品表面能看到【{absent}】的印花吗？", f"否，图中不存在【{absent}】。"))

    elif case == "occluded":
        zone = rec["e_zone_zh"]
        items.extend([
            (f"请判断：这件商品上是否印有【{target}】图案？",
             f"无法确认：{zone}被同色面料/遮挡物覆盖，未暴露可辨识的图案。"),
            (f"画面中是否有证据支持【{target}】的存在？",
             "没有可辨识证据：目标印制区被完全遮挡，无法观察有效图案。"),
            (f"能否根据当前画面断定【{target}】不存在？",
             "不能：目标印制区被遮挡，无法得出否定结论。"),
            ("该商品的印花印制区是否完整可见？",
             f"否：{zone}被同色面料/遮挡物覆盖。"),
            (f"请描述【{target}】在该商品上的可见情况。",
             f"不可见：{zone}被遮挡，无法确认是否印有该图案。"),
            ("图中能看到哪些可辨识的印花图案？",
             "画面中未见可辨识的印花图案；目标印制区被遮挡，无法确认其内容。"),
            ("该图案是否在画面中清晰可见？", "不可见：目标印制区被完全遮挡。"),
            ("图中商品属于什么品类？", cat),
        ])

    elif case == "hard_distractor":
        actual = rec.get("e_actual") or {}
        actual_name = display_name(actual) if actual else ""
        actual_bbox = fmt_bbox(actual["bbox"]) if actual else ""
        actual_short = (actual.get("name_zh") or actual.get("display_short") or "") if actual else ""
        items.extend([
            (f"图中商品上是否存在【{target}】图案？", f"否，图中不存在【{target}】。"),
            (f"请确认商品上的图案是否为【{target}】？",
             f"不是：图中实际图案为【{actual_name}】，与【{target}】不同。"),
            (f"图中实际印制的图案名称是什么？", actual_name),
            ("请识别商品上实际出现的图案。", actual_name),
            ("图中实际图案的归一化边界框是多少？", actual_bbox),
            ("请定位图中实际出现的图案并输出 [ymin,xmin,ymax,xmax]。", actual_bbox),
            (f"图中图案与【{target}】是否相同？",
             f"不同：图中实际图案为【{actual_name}】，并非【{target}】。"),
            ("图中共有几处独立印花？", "共有 1 处独立印花/视觉元素。"),
            ("图中商品属于什么品类？", cat),
        ])
        if actual_short:
            items.append((f"图中实际出现的图案是否为「{actual_short}」？", f"是，实际图案为【{actual_name}】。"))
        if cate_group:
            items.append((CATEGORY_IDENTIFY_Q[cate], actual_name))
        if actual.get("color"):
            items.append(("图中实际图案的代表色是什么？", actual["color"]))
        for absent in pick_absent_entities(rec, rng, k=2, prefer_domains=cate_group):
            items.append((f"图中商品上是否存在【{absent}】图案？", f"否，图中不存在【{absent}】。"))
            items.append((f"商品表面能看到【{absent}】的印花吗？", f"否，图中不存在【{absent}】。"))
    return dedupe_items(items)


def items_for(rec: Dict[str, Any], cate: str, seed: int) -> List[Tuple[str, str]]:
    rng = rng_for(seed, "items", cate, rec["v2_sample_id"])
    if rec["stage"] in ("b", "c"):
        return grounding_items(rec, cate, rng)
    return negative_items(rec, cate, rng)


# ---------------------------------------------------------------------------
# 行构造
# ---------------------------------------------------------------------------
def make_multi_turn_row(rec: Dict[str, Any],
                        cate: str,
                        items: Sequence[Tuple[str, str]],
                        rounds: int,
                        view: int,
                        attempt: int,
                        seed: int,
                        used_anchors: Sequence[str] = (),
                        required: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    pick_rng = rng_for(seed, "pick", cate, rec["v2_sample_id"], view, attempt)
    chosen = select_round_items(items, rounds, pick_rng, view, used_anchors, required)
    messages: List[Dict[str, str]] = []
    for idx, (question, answer) in enumerate(chosen):
        content = ("<image>\n" + question) if idx == 0 else question
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": answer})
    messages = trim_messages_to_limit(messages)
    rule = f"stage_{rec['stage']}" if rec["stage"] in ("b", "c") else f"stage_e_{rec['e_case']}"
    return {
        "row_type": "multi_turn_qa",
        "cate_id": cate,
        "cate_rule": rule,
        "v2_sample_id": rec["v2_sample_id"],
        "v2_stage": rec["stage"],
        "images": [rec["image_path"]],
        "messages": messages,
        "qa_rounds": len(messages) // 2,
        "view_index": view,
    }


def _consistent_templates(rec: Dict[str, Any]) -> List[Dict[str, str]]:
    cat = rec["category_zh"]
    name = display_name(rec["entities"][0])
    bbox = fmt_bbox(rec["entities"][0]["bbox"])
    color = rec["color_answer"] or rec["entities"][0]["color"]
    location = rec.get("location_answer")
    tech = rec["technique"]
    out: List[Dict[str, str]] = []
    if len(rec["entities"]) >= 2:
        first, second = rec["entities"][0], rec["entities"][1]
        name1, name2 = display_name(first), display_name(second)
        bbox1, bbox2 = fmt_bbox(first["bbox"]), fmt_bbox(second["bbox"])
        rel = rec.get("relation_answer") or ""
        count = rec["count"]
        count_disp = f"共有 {count} 处独立印花/视觉元素。"
        out.extend([
            {"statement": f"图中共有 {count} 处独立印花/视觉元素，主视觉图案为【{name1}】，"
                          f"次要图案为【{name2}】。",
             "evidence": "图案数量与主/次图元名称均与图片一致。",
             "follow_q": "图中共有几处独立印花？", "follow_a": count_disp},
            {"statement": f"主视觉图案【{name1}】与次要图案【{name2}】的相对方位为{rel}。",
             "evidence": "两图元中心点的相对方位与描述一致。",
             "follow_q": "次要图案相对主视觉图案位于哪个方位？", "follow_a": rel},
            {"statement": f"主视觉图案【{name1}】的边界框为 {bbox1}，次要图案【{name2}】的边界框为 {bbox2}。",
             "evidence": "两个图元的边界框均与图片一致。",
             "follow_q": "次要图案的归一化边界框是多少？", "follow_a": bbox2},
            {"statement": f"图中共有 {count} 处独立印花，主视觉图案【{name1}】与次要图案"
                          f"【{name2}】的相对方位为{rel}。",
             "evidence": "印花数量与相对方位均与图片一致。",
             "follow_q": "主视觉图案与次要图案的相对位置关系是什么？", "follow_a": rel},
            {"statement": f"次要图案【{name2}】的代表色为{color}，与主视觉图案【{name1}】"
                          f"的相对方位为{rel}。",
             "evidence": "次要图案代表色与两图元相对方位均与记录一致。",
             "follow_q": "主视觉图案与次要图案的相对位置关系是什么？", "follow_a": rel},
        ])
    else:
        if location:
            out.append({
                "statement": f"{cat}上的主视觉图案【{name}】位于{location}，印花采用{tech}工艺。",
                "evidence": "商品品类、图案位置与印花工艺均与样本记录一致。",
                "follow_q": "该图案实际位于商品的哪个位置？", "follow_a": location})
            out.append({
                "statement": f"图中商品为{cat}，主视觉图案【{name}】位于{location}。",
                "evidence": "商品品类与图案工艺位置均与图片一致。",
                "follow_q": "该图案位于商品的哪个位置？", "follow_a": location})
            out.append({
                "statement": f"主视觉图案【{name}】位于{location}，其代表色为{color}。",
                "evidence": "图案位置与代表色均与记录一致。",
                "follow_q": "该图案的代表色是什么？", "follow_a": color})
        out.extend([
            {"statement": f"主视觉图案【{name}】的代表色为{color}，印花采用{tech}工艺。",
             "evidence": "图案代表色与印花工艺均取自该样本的真实记录。",
             "follow_q": "该图案的代表色是什么？", "follow_a": color},
            {"statement": f"图中商品为{cat}，主视觉图案【{name}】的归一化边界框为 {bbox}。",
             "evidence": "商品品类与图案边界框均与图片一致。",
             "follow_q": "主视觉图案的归一化边界框是多少？", "follow_a": bbox},
            {"statement": f"主视觉图案【{name}】代表色为{color}，其归一化边界框为 {bbox}。",
             "evidence": "代表色与边界框均与图片一致。",
             "follow_q": "主视觉图案的归一化边界框是多少？", "follow_a": bbox},
        ])
    return out


def _inconsistent_templates(rec: Dict[str, Any], rng: random.Random) -> List[Dict[str, str]]:
    cat = rec["category_zh"]
    tech = rec["technique"]
    out: List[Dict[str, str]] = []
    if len(rec["entities"]) >= 2:
        first, second = rec["entities"][0], rec["entities"][1]
        name1, name2 = display_name(first), display_name(second)
        bbox1, bbox2 = fmt_bbox(first["bbox"]), fmt_bbox(second["bbox"])
        color2 = second["color"]
        rel = rec.get("relation_answer") or ""
        count = rec["count"]
        # 1) 次图元颜色篡改
        fake_color = pick_other(COLOR_VOCAB, color2, rng)
        if fake_color:
            out.append({
                "statement": f"次要图案【{name2}】的代表色为{fake_color}，"
                             f"图中共有 {count} 处独立印花/视觉元素。",
                "aspect_zh": "颜色", "fake_disp": f"「{fake_color}」",
                "true_disp": f"「{color2}」",
                "evidence": "次要图案代表色取自资产调色板记录。",
                "follow_q": "次要图案真实的代表色是什么？", "follow_a": color2})
        # 2) 工艺篡改
        fake_tech = pick_other(TECH_VOCAB, tech, rng)
        if fake_tech:
            out.append({
                "statement": f"{cat}上的两处印花均采用{fake_tech}工艺。",
                "aspect_zh": "印花工艺", "fake_disp": f"「{fake_tech}」",
                "true_disp": f"「{tech}」",
                "evidence": "印花工艺以该商品底板的真实记录为准。",
                "follow_q": "该商品实际采用的印花工艺是什么？", "follow_a": tech})
        # 3) 主图元 bbox 篡改
        fake_bbox1 = fmt_bbox(shift_bbox(first["bbox"], rng))
        out.append({
            "statement": f"主视觉图案【{name1}】的归一化边界框为 {fake_bbox1}，"
                         f"次要图案为【{name2}】。",
            "aspect_zh": "边界框", "fake_disp": fake_bbox1, "true_disp": bbox1,
            "evidence": "边界框由形变后的可见图案区域反算。",
            "follow_q": "请给出主视觉图案真实的归一化边界框。", "follow_a": bbox1})
        # 4) 次图元 bbox 篡改
        fake_bbox2 = fmt_bbox(shift_bbox(second["bbox"], rng))
        out.append({
            "statement": f"次要图案【{name2}】的归一化边界框为 {fake_bbox2}。",
            "aspect_zh": "边界框", "fake_disp": fake_bbox2, "true_disp": bbox2,
            "evidence": "边界框由形变后的可见图案区域反算。",
            "follow_q": "请给出次要图案真实的归一化边界框。", "follow_a": bbox2})
        # 5) 数量篡改（真值恒为 2，假值 3）
        fake_count = 3 if count != 3 else 4
        fake_count_disp = f"共有 {fake_count} 处独立印花/视觉元素"
        true_count_disp = f"共有 {count} 处独立印花/视觉元素"
        out.append({
            "statement": f"图中共有 {fake_count} 处独立印花/视觉元素，主视觉图案为【{name1}】。",
            "aspect_zh": "印花数量", "fake_disp": fake_count_disp, "true_disp": true_count_disp,
            "evidence": "按互不重叠的图元实例逐一计数。",
            "follow_q": "图中共有几处独立印花？",
            "follow_a": rec.get("count_answer") or f"共有 {count} 处独立印花/视觉元素。"})
        # 6) 相对方位篡改
        fake_rel = rng.choice([r for r in RELATION_FAKE if r != rel])
        out.append({
            "statement": f"主视觉图案【{name1}】与次要图案【{name2}】的相对方位为{fake_rel}。",
            "aspect_zh": "相对方位", "fake_disp": f"「{fake_rel}」", "true_disp": f"「{rel}」",
            "evidence": "比较两图元边界框中心点得到相对方位。",
            "follow_q": "次要图案相对主视觉图案的实际方位是什么？", "follow_a": rel})
    else:
        name = display_name(rec["entities"][0])
        bbox = fmt_bbox(rec["entities"][0]["bbox"])
        color = rec["color_answer"] or rec["entities"][0]["color"]
        # 1) 颜色篡改
        fake_color = pick_other(COLOR_VOCAB, color, rng)
        if fake_color:
            out.append({
                "statement": f"主视觉图案【{name}】的代表色为{fake_color}，印花采用{tech}工艺。",
                "aspect_zh": "颜色", "fake_disp": f"「{fake_color}」",
                "true_disp": f"「{color}」",
                "evidence": "图案代表色取自资产调色板记录。",
                "follow_q": "该图案真实的代表色是什么？", "follow_a": color})
        # 2) 工艺篡改
        fake_tech = pick_other(TECH_VOCAB, tech, rng)
        if fake_tech:
            out.append({
                "statement": f"{cat}上的主视觉图案【{name}】采用{fake_tech}工艺。",
                "aspect_zh": "印花工艺", "fake_disp": f"「{fake_tech}」",
                "true_disp": f"「{tech}」",
                "evidence": "印花工艺以该商品底板的真实记录为准。",
                "follow_q": "该商品实际采用的印花工艺是什么？", "follow_a": tech})
        # 3) 工艺位置篡改（跨品类区位词，确保与真值不同）
        fake_position = pick_other_position(rec, rng)
        if fake_position:
            fake_pos, true_pos = fake_position
            out.append({
                "statement": f"主视觉图案【{name}】位于{fake_pos}。",
                "aspect_zh": "工艺位置", "fake_disp": f"「{fake_pos}」",
                "true_disp": f"「{true_pos}」",
                "evidence": "该图案的工艺位置由商品底板与贴附区域确定。",
                "follow_q": "该图案实际位于商品的哪个位置？", "follow_a": true_pos})
        # 4) bbox 篡改
        fake_bbox = fmt_bbox(shift_bbox(rec["entities"][0]["bbox"], rng))
        out.append({
            "statement": f"主视觉图案【{name}】的归一化边界框为 {fake_bbox}。",
            "aspect_zh": "边界框", "fake_disp": fake_bbox, "true_disp": bbox,
            "evidence": "边界框由形变后的可见图案区域反算。",
            "follow_q": "请给出该图案真实的归一化边界框。", "follow_a": bbox})
    return out


def make_compliance_row(rec: Dict[str, Any],
                        label: str,
                        two_turns: bool,
                        attempt: int,
                        seed: int) -> Dict[str, Any]:
    build_rng = rng_for(seed, "compliance", rec["v2_sample_id"], attempt)
    if label == "consistent":
        template = build_rng.choice(_consistent_templates(rec))
        answer = f"一致。依据：{template['evidence']}"
        check = {"label": "consistent"}
    else:
        template = build_rng.choice(_inconsistent_templates(rec, build_rng))
        answer = (f"不一致：描述中的{template['aspect_zh']}与图片不符"
                  f"（正确应为{template['true_disp']}）。依据：{template['evidence']}")
        check = {"label": "inconsistent", "aspect": template["aspect_zh"],
                 "fake": template["fake_disp"], "true": template["true_disp"]}
    prompt = COMPLIANCE_PROMPTS[build_rng.randrange(len(COMPLIANCE_PROMPTS))]
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "<image>\n" + prompt.format(statement=template["statement"])},
        {"role": "assistant", "content": answer},
    ]
    if two_turns:
        messages.append({"role": "user", "content": template["follow_q"]})
        messages.append({"role": "assistant", "content": template["follow_a"]})
    messages = trim_messages_to_limit(messages)
    return {
        "row_type": "text_compliance",
        "cate_id": "text-compliance",
        "cate_rule": label,
        "v2_sample_id": rec["v2_sample_id"],
        "v2_stage": rec["stage"],
        "images": [rec["image_path"]],
        "messages": messages,
        "qa_rounds": len(messages) // 2,
        "compliance_label": label,
        "compliance_check": check,
        "statement": template["statement"],
    }


def caption_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    elements = []
    for entity in rec["entities"]:
        elements.append({
            "name_zh": display_name(entity),
            "bbox_1000": [int(v) for v in entity["bbox"]],
            "color_zh": entity["color"],
        })
    return {
        "category_zh": rec["category_zh"],
        "scene_style_zh": rec["scene_style_zh"],
        "placement_zone": rec["placement_zone"],
        "print_technique_zh": rec["technique"],
        "elements": elements,
        "caption_zh": rec["caption_zh"],
    }


def make_caption_row(rec: Dict[str, Any], template_idx: int) -> Dict[str, Any]:
    payload = caption_payload(rec)
    assistant = json.dumps(payload, ensure_ascii=False)
    messages = [
        {"role": "user", "content": "<image>\n" + CAPTION_PROMPTS[template_idx]},
        {"role": "assistant", "content": assistant},
    ]
    messages = trim_messages_to_limit(messages)
    return {
        "row_type": "structured_caption",
        "cate_id": "structured_caption",
        "cate_rule": f"stage_{rec['stage']}",
        "v2_sample_id": rec["v2_sample_id"],
        "v2_stage": rec["stage"],
        "images": [rec["image_path"]],
        "messages": messages,
        "qa_rounds": 1,
        "caption_template": int(template_idx),
        "caption_payload": payload,
    }


# ---------------------------------------------------------------------------
# 内容注册表（防止完全重复行）
# ---------------------------------------------------------------------------
def row_signature(messages: Sequence[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple((str(m["role"]), str(m["content"])) for m in messages)


def commit_row(builder, registry: Dict[str, Any], records: List[Dict[str, Any]],
               max_attempts: int = 16) -> None:
    """尝试以不同 attempt 构造内容唯一行；极端情况下接受重复并计数（不中断生成）。"""
    candidate = None
    for attempt in range(max_attempts):
        row = builder(attempt)
        if row is None:
            continue
        candidate = row
        signature = row_signature(row["messages"])
        if signature not in registry["global"]:
            registry["global"].add(signature)
            records.append(row)
            return
    if candidate is None:
        raise RuntimeError("行构造均返回 None")
    registry["duplicates"] += 1
    records.append(candidate)


# ---------------------------------------------------------------------------
# 各 cate 构建
# ---------------------------------------------------------------------------
def build_multi_turn_cate(cate: str,
                          pool: Sequence[Dict[str, Any]],
                          target: int,
                          seed: int,
                          records: List[Dict[str, Any]],
                          registry: Dict[str, Any],
                          build_stats: Dict[str, Any]) -> None:
    pool = sorted(pool, key=lambda rec: rec["v2_sample_id"])
    counts = plan_views(target, len(pool), MAX_VIEWS[cate], cate)
    build_stats[cate] = {
        "pool_size": len(pool),
        "rows_target": target,
        "view_count_dist": dict(sorted(Counter(counts).items())),
        "source_stage_dist": dict(sorted(Counter(rec["stage"] for rec in pool).items())),
        "source_rule_dist": dict(sorted(Counter(
            (f"stage_{rec['stage']}" if rec["stage"] in ("b", "c")
             else f"stage_e_{rec['e_case']}") for rec in pool).items())),
    }
    for rec, n_views in zip(pool, counts):
        items = items_for(rec, cate, seed)
        if len(items) < 2:
            raise RuntimeError(f"{cate} {rec['v2_sample_id']} 可用问答条目不足：{len(items)}")
        # cate 关键能力保障：quantity 必须落在计数问答上；misleading 必须包含对目标的
        # 否定/证据不足问答（e 记录的首条 QA 恒为 yes_no 否定或 uncertain）
        required: List[Tuple[str, str]] = []
        if cate in ("quantity", "misleading") and rec["qa"]:
            required.append((rec["qa"][0]["question_zh"], rec["qa"][0]["answer_zh"]))
        # hard_distractor：除「询问原目标→否」外，强制再问一次图中真实图案（答案含名称+bbox）
        if (cate == "misleading" and rec.get("e_case") == "hard_distractor"
                and len(rec["qa"]) >= 2):
            required.append((rec["qa"][1]["question_zh"], rec["qa"][1]["answer_zh"]))
        min_rounds = ROUND_RANGES[cate][0]
        if cate == "misleading" and rec.get("e_case") == "hard_distractor":
            min_rounds = 3
        used_anchors: List[str] = []
        for view in range(n_views):
            round_rng = rng_for(seed, "rounds", cate, rec["v2_sample_id"], view)
            rounds = max(min_rounds, choose_rounds(cate, round_rng, len(items)))
            rounds = min(rounds, ROUND_RANGES[cate][1])

            max_rounds = ROUND_RANGES[cate][1]

            def builder(attempt: int, rec=rec, cate=cate, items=items, rounds=rounds,
                        view=view, used_anchors=tuple(used_anchors),
                        required=tuple(required), max_rounds=max_rounds) -> Dict[str, Any]:
                # 每 4 次 attempt 允许增加一轮（去重手段之一，仍在 cate 轮数上限内）
                effective_rounds = min(rounds + attempt // 4, max_rounds)
                return make_multi_turn_row(rec, cate, items, effective_rounds, view, attempt,
                                           seed, used_anchors, required)

            row = builder(0)
            first_user = str(row["messages"][0]["content"])
            used_anchors.append(first_user.split("\n", 1)[-1] if "\n" in first_user else first_user)
            commit_row(builder, registry, records)


def build_compliance_cate(pool: Sequence[Dict[str, Any]],
                          target: int,
                          seed: int,
                          records: List[Dict[str, Any]],
                          registry: Dict[str, Any],
                          build_stats: Dict[str, Any]) -> None:
    pool = sorted(pool, key=lambda rec: rec["v2_sample_id"])
    selected = select_evenly(pool, target)
    label_rng = rng_for(seed, "compliance-labels")
    order = list(range(len(selected)))
    label_rng.shuffle(order)
    half = target // 2
    consistent_indices = set(order[:half])
    label_dist: Counter = Counter()
    for index, rec in enumerate(selected):
        label = "consistent" if index in consistent_indices else "inconsistent"
        label_dist[label] += 1
        two_turns = rng_for(seed, "compliance-rounds", rec["v2_sample_id"]).random() < 0.45

        def builder(attempt: int, rec=rec, label=label, two_turns=two_turns) -> Dict[str, Any]:
            return make_compliance_row(rec, label, two_turns, attempt, seed)

        commit_row(builder, registry, records)
    build_stats["text-compliance"] = {
        "pool_size": len(pool),
        "rows_target": target,
        "selected_records": len(selected),
        "label_dist": dict(label_dist),
        "source_stage_dist": dict(sorted(Counter(rec["stage"] for rec in selected).items())),
    }


def build_caption_cate(pool: Sequence[Dict[str, Any]],
                       target: int,
                       seed: int,
                       records: List[Dict[str, Any]],
                       registry: Dict[str, Any],
                       build_stats: Dict[str, Any]) -> None:
    pool = sorted(pool, key=lambda rec: rec["v2_sample_id"])
    counts = plan_views(target, len(pool), MAX_VIEWS["structured_caption"], "structured_caption")
    build_stats["structured_caption"] = {
        "pool_size": len(pool),
        "rows_target": target,
        "view_count_dist": dict(sorted(Counter(counts).items())),
        "source_stage_dist": dict(sorted(Counter(rec["stage"] for rec in pool).items())),
    }
    template_dist: Counter = Counter()
    for index, (rec, n_views) in enumerate(zip(pool, counts)):
        for view in range(n_views):
            base_template = (index + view) % 2

            def builder(attempt: int, rec=rec, base_template=base_template) -> Dict[str, Any]:
                # 若与已有行完全重复，换另一种指令模板重试
                return make_caption_row(rec, (base_template + attempt) % 2)

            candidate = None
            for attempt in range(2):
                row = builder(attempt)
                candidate = row
                signature = row_signature(row["messages"])
                if signature not in registry["global"]:
                    registry["global"].add(signature)
                    records.append(row)
                    template_dist[(base_template + attempt) % 2] += 1
                    break
            else:
                registry["duplicates"] += 1
                records.append(candidate)
                template_dist[base_template] += 1
    build_stats["structured_caption"]["template_dist"] = dict(sorted(template_dist.items()))


def build_all(records_pool: Sequence[Dict[str, Any]],
              seed: int,
              build_stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    registry: Dict[str, Any] = {"global": set(), "duplicates": 0}

    bc_records = [rec for rec in records_pool if rec["stage"] in ("b", "c")]
    e_records = [rec for rec in records_pool if rec["stage"] == "e"]
    bce_negative_records = [rec for rec in e_records
                            if rec["e_case"] in ("absent_clean", "hard_distractor")]

    def domain_pool(groups: Sequence[str]) -> List[Dict[str, Any]]:
        bc = [rec for rec in bc_records if any(k in rec["domain"] for k in groups)]
        e = [rec for rec in bce_negative_records if any(k in rec["domain"] for k in groups)]
        return bc + e

    apparel_pool = [rec for rec in bc_records if rec["is_apparel"]]
    missing_apparel = len(bc_records) - len(apparel_pool)
    if missing_apparel:
        # b/c 全部应为服装/箱包/鞋履/帽饰底板；允许极少数词表未命中的品类仍计入 apparel
        apparel_pool = bc_records
    build_stats.setdefault("_pools", {})["apparel_missing_keyword"] = missing_apparel
    build_stats["_pools"]["bc_records"] = len(bc_records)
    build_stats["_pools"]["e_records"] = len(e_records)
    build_stats["_pools"]["e_absent_distractor"] = len(bce_negative_records)

    build_multi_turn_cate("apparel", apparel_pool, CATE_TARGETS["apparel"], seed,
                          records, registry, build_stats)
    build_multi_turn_cate("flag-hard", domain_pool(OFFICIAL_DOMAINS),
                          CATE_TARGETS["flag-hard"], seed, records, registry, build_stats)
    build_multi_turn_cate("ip-brand", domain_pool(BRAND_DOMAINS),
                          CATE_TARGETS["ip-brand"], seed, records, registry, build_stats)
    build_multi_turn_cate("anime-ip", domain_pool(ANIME_DOMAINS),
                          CATE_TARGETS["anime-ip"], seed, records, registry, build_stats)
    build_multi_turn_cate("misleading", e_records,
                          CATE_TARGETS["misleading"], seed, records, registry, build_stats)
    quantity_pool = [rec for rec in bc_records
                     if rec["stage"] == "c" and rec["count_answer"]
                     and ({"count", "relation"} <= {q.get("answer_type") for q in rec["qa"]})]
    build_multi_turn_cate("quantity", quantity_pool,
                          CATE_TARGETS["quantity"], seed, records, registry, build_stats)
    build_compliance_cate(bc_records, CATE_TARGETS["text-compliance"], seed,
                          records, registry, build_stats)
    build_caption_cate(bc_records, CATE_TARGETS["structured_caption"], seed,
                       records, registry, build_stats)

    # 分配 sample_id（每 cate 独立递增，格式 V6B_S_{cate}_{序号:06d}）
    counters: Counter = Counter()
    for record in records:
        cate = record["cate_id"]
        counters[cate] += 1
        record["sample_id"] = f"V6B_S_{cate.replace('-', '_')}_{counters[cate]:06d}"
    build_stats["duplicate_content_accepts"] = registry["duplicates"]
    return records, build_stats


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
ROUND_RANGES = {
    "apparel": (2, 5),
    "flag-hard": (2, 5),
    "ip-brand": (2, 5),
    "anime-ip": (2, 5),
    "misleading": (2, 4),
    "quantity": (3, 5),
    "text-compliance": (1, 2),
    "structured_caption": (1, 1),
}
REQUIRED_CAPTION_FIELDS = {"category_zh", "scene_style_zh", "placement_zone",
                           "print_technique_zh", "elements", "caption_zh"}


def stats_summary(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


def validate_row(record: Dict[str, Any], image_exists: Dict[str, bool]) -> None:
    sample_id = record["sample_id"]
    messages = record["messages"]
    images = record["images"]
    if not messages or len(messages) % 2 != 0:
        raise AssertionError(f"{sample_id}: messages 非 user/assistant 交替结构")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise AssertionError(f"{sample_id}: 第 {index} 条 role 应为 {expected}")
        if not str(message.get("content") or "").strip():
            raise AssertionError(f"{sample_id}: 第 {index} 条 content 为空")
    if len(images) != 1:
        raise AssertionError(f"{sample_id}: 单图行 images 长度应为 1")
    placeholder_total = sum(str(m["content"]).count("<image>") for m in messages)
    placeholder_first = str(messages[0]["content"]).count("<image>")
    if placeholder_total != 1 or placeholder_first != 1:
        raise AssertionError(f"{sample_id}: <image> 数量/位置不正确")
    if not image_exists.get(images[0]):
        raise AssertionError(f"{sample_id}: 图片不存在 {images[0]}")
    if row_token_estimate(messages, len(images)) > MAX_LEN:
        raise AssertionError(f"{sample_id}: 估算长度超 {MAX_LEN}")
    low, high = ROUND_RANGES[record["cate_id"]]
    if not (low <= record["qa_rounds"] <= high):
        raise AssertionError(f"{sample_id}: rounds={record['qa_rounds']} 不在 [{low},{high}]")
    if record["row_type"] == "text_compliance":
        check = record["compliance_check"]
        if check["label"] == "inconsistent":
            if check["fake"] == check["true"]:
                raise AssertionError(f"{sample_id}: 篡改值与真值相同")
            if check["true"] not in record["messages"][1]["content"]:
                raise AssertionError(f"{sample_id}: 不一致答案未包含真值")
        if record["qa_rounds"] not in (1, 2):
            raise AssertionError(f"{sample_id}: compliance 轮数异常")
    elif record["row_type"] == "structured_caption":
        payload = json.loads(record["messages"][1]["content"])
        if set(payload) != REQUIRED_CAPTION_FIELDS:
            raise AssertionError(f"{sample_id}: caption 字段不符 {sorted(payload)}")
        for element in payload["elements"]:
            if set(element) != {"name_zh", "bbox_1000", "color_zh"}:
                raise AssertionError(f"{sample_id}: element 字段不符")
            bbox = element["bbox_1000"]
            if not (0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000):
                raise AssertionError(f"{sample_id}: element bbox 非法 {bbox}")
            if not element["name_zh"] or not element["color_zh"]:
                raise AssertionError(f"{sample_id}: element 名称/颜色为空")
        if not payload["category_zh"] or not payload["caption_zh"]:
            raise AssertionError(f"{sample_id}: caption 品类/描述为空")


def self_check(records: Sequence[Dict[str, Any]],
               image_exists: Dict[str, bool]) -> Dict[str, Any]:
    if len(records) != TOTAL_ROWS:
        raise AssertionError(f"总行数 {len(records)} != {TOTAL_ROWS}")
    cate_counter = Counter(record["cate_id"] for record in records)
    for cate in CATE_ORDER:
        if cate_counter.get(cate, 0) != CATE_TARGETS[cate]:
            raise AssertionError(
                f"cate={cate} 行数 {cate_counter.get(cate, 0)} != 配额 {CATE_TARGETS[cate]}")
    sample_ids, row_keys = set(), set()
    cate_image_signatures: Dict[Tuple[str, str], set] = defaultdict(set)
    for record in records:
        validate_row(record, image_exists)
        sample_id = record["sample_id"]
        if sample_id in sample_ids:
            raise AssertionError(f"sample_id 重复：{sample_id}")
        sample_ids.add(sample_id)
        key = (sample_id, record["cate_id"], tuple(record["images"]))
        if key in row_keys:
            raise AssertionError(f"行级键重复：{key}")
        row_keys.add(key)
        message_sig = row_signature(record["messages"])
        image_key = (record["cate_id"], record["images"][0])
        if message_sig in cate_image_signatures[image_key]:
            raise AssertionError(f"{sample_id}: 同一 (cate, 图片) 下出现完全相同的行内容")
        cate_image_signatures[image_key].add(message_sig)
    global_signatures = Counter(row_signature(record["messages"]) for record in records)
    exact_duplicate_rows = sum(count for count in global_signatures.values() if count > 1)
    return {
        "cate_dist": {cate: cate_counter[cate] for cate in CATE_ORDER},
        "cate_exact_match": True,
        "sample_id_unique": True,
        "row_key_unique": True,
        "image_sequence_unique_per_cate_image": True,
        "exact_duplicate_content_rows": exact_duplicate_rows,
        "exact_duplicate_content_groups": sum(1 for count in global_signatures.values() if count > 1),
        "max_rows_per_cate_image": max(
            len(signatures) for signatures in cate_image_signatures.values()),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def truncate(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def build_manifest(records: Sequence[Dict[str, Any]],
                   check: Dict[str, Any],
                   build_stats: Dict[str, Any],
                   source_stats: Dict[str, Any],
                   dataset_path: Path,
                   out_parquet: Path,
                   seed: int) -> Dict[str, Any]:
    row_type_dist = Counter(record["row_type"] for record in records)
    cate_dist = {cate: sum(1 for r in records if r["cate_id"] == cate) for cate in CATE_ORDER}
    cate_rule_dist: Dict[str, Dict[str, int]] = {}
    for record in records:
        cate_rule_dist.setdefault(record["cate_id"], Counter())[record["cate_rule"]] += 1
    stage_crosstab: Dict[str, Dict[str, int]] = {}
    for record in records:
        stage_crosstab.setdefault(record["cate_id"], Counter())[record["v2_stage"]] += 1
    rounds_dist: Counter = Counter(record["qa_rounds"] for record in records)
    rounds_per_cate: Dict[str, Dict[str, int]] = {}
    for record in records:
        rounds_per_cate.setdefault(record["cate_id"], Counter())[record["qa_rounds"]] += 1
    view_per_cate_image = Counter((record["cate_id"], record["images"][0]) for record in records)
    image_views_dist: Dict[str, Dict[str, int]] = {}
    for cate in CATE_ORDER:
        counter = Counter(count for (c, _), count in view_per_cate_image.items() if c == cate)
        image_views_dist[cate] = {str(k): v for k, v in sorted(counter.items())}

    all_images = [image for record in records for image in record["images"]]
    image_counts = Counter(all_images)
    reuse_counts = Counter(image_counts.values())
    text_tokens: List[float] = []
    total_tokens: List[float] = []
    text_stats_per_cate: Dict[str, Dict[str, Any]] = {}
    text_stats_per_type: Dict[str, Dict[str, Any]] = {}
    for group in (CATE_ORDER, ("multi_turn_qa", "text_compliance", "structured_caption")):
        for key in group:
            if key in CATE_ORDER:
                subset = [record for record in records if record["cate_id"] == key]
            else:
                subset = [record for record in records if record["row_type"] == key]
            tokens = [estimate_text_tokens(messages_text(record["messages"])) for record in subset]
            totals = [token + IMAGE_TOKEN_ESTIMATE for token in tokens]
            stats = {
                "rows": len(subset),
                "text_tokens_est": stats_summary(tokens),
                "tokens_est_incl_image_512": stats_summary(totals),
                "text_tokens_est_total": round(sum(tokens), 1),
            }
            if key in CATE_ORDER:
                text_stats_per_cate[key] = stats
            else:
                text_stats_per_type[key] = stats
    for record in records:
        tokens = estimate_text_tokens(messages_text(record["messages"]))
        text_tokens.append(tokens)
        total_tokens.append(tokens + IMAGE_TOKEN_ESTIMATE)

    compliance_label_dist = dict(sorted(Counter(
        record["compliance_label"] for record in records
        if record["row_type"] == "text_compliance").items()))
    compliance_aspect_dist = dict(sorted(Counter(
        record["compliance_check"].get("aspect", "consistent")
        for record in records if record["row_type"] == "text_compliance").items()))
    caption_template_dist = dict(sorted(Counter(
        record["caption_template"] for record in records
        if record["row_type"] == "structured_caption").items()))

    def example(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sample_id": record["sample_id"],
            "cate_id": record["cate_id"],
            "row_type": record["row_type"],
            "v2_sample_id": record["v2_sample_id"],
            "v2_stage": record["v2_stage"],
            "images": record["images"],
            "qa_rounds": record["qa_rounds"],
            "messages": [{"role": message["role"], "content": truncate(message["content"])}
                         for message in record["messages"]],
        }

    examples = []
    for row_type in ("multi_turn_qa", "text_compliance", "structured_caption"):
        record = next((r for r in records if r["row_type"] == row_type), None)
        if record is not None:
            examples.append(example(record))

    manifest: Dict[str, Any] = {
        "part": "single_rows",
        "rows": len(records),
        "variant": VARIANT,
        "seed": seed,
        "max_len": MAX_LEN,
        "image_token_estimate": IMAGE_TOKEN_ESTIMATE,
        "split_policy": "train_only",
        "quota_targets": dict(CATE_TARGETS),
        "quota_total": TOTAL_ROWS,
        "cate_dist": cate_dist,
        "cate_exact_match": check["cate_exact_match"],
        "row_type_dist": dict(sorted(row_type_dist.items())),
        "cate_rule_dist": {cate: dict(sorted(counter.items()))
                           for cate, counter in sorted(cate_rule_dist.items())},
        "v2_stage_crosstab": {cate: dict(sorted(counter.items()))
                              for cate, counter in sorted(stage_crosstab.items())},
        "qa_round_dist": {str(k): v for k, v in sorted(rounds_dist.items())},
        "qa_round_dist_per_cate": {cate: {str(k): v for k, v in sorted(counter.items())}
                                   for cate, counter in sorted(rounds_per_cate.items())},
        "compliance_label_dist": compliance_label_dist,
        "compliance_aspect_dist": compliance_aspect_dist,
        "caption_template_dist": caption_template_dist,
        "multi_view": {
            "rows_per_image_dist_per_cate": image_views_dist,
            "max_rows_for_one_image_in_one_cate": max(view_per_cate_image.values()),
        },
        "image_count_dist": {"1": len(records)},
        "images": {
            "total_references": len(all_images),
            "unique_paths": len(image_counts),
            "reused_paths": len(image_counts) - sum(1 for c in image_counts.values() if c == 1),
            "reuse_count_dist": {str(k): v for k, v in sorted(reuse_counts.items())},
            "max_reuse": max(image_counts.values()),
            "all_files_exist": all((PROJECT_ROOT / image).is_file() for image in image_counts),
        },
        "unique_v2_records_used": len({record["v2_sample_id"] for record in records}),
        "uniqueness": {
            "sample_id_unique": check["sample_id_unique"],
            "row_key_sample_id_cate_images_unique": check["row_key_unique"],
            "image_sequence_unique_per_cate_image": check["image_sequence_unique_per_cate_image"],
            "exact_duplicate_content_rows": check["exact_duplicate_content_rows"],
            "exact_duplicate_content_groups": check["exact_duplicate_content_groups"],
            "max_rows_per_cate_image": check["max_rows_per_cate_image"],
        },
        "text_stats_all": {
            "rows": len(records),
            "text_tokens_est": stats_summary(text_tokens),
            "text_tokens_est_total": round(sum(text_tokens), 1),
            "tokens_est_incl_image_512": stats_summary(total_tokens),
            "tokens_est_total": round(sum(total_tokens), 1),
            "over_max_len": sum(1 for value in total_tokens if value > MAX_LEN),
        },
        "text_stats_per_cate": text_stats_per_cate,
        "text_stats_per_row_type": text_stats_per_type,
        "cate_build": build_stats,
        "source": {
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "stats": source_stats,
            "filters": {
                "split": ["train"],
                "task_stage": list(STAGE_BCE),
                "image_exists": True,
                "cate_pools": {
                    "apparel": "stage_b/c，底板为服装/箱包/鞋履/帽饰（b/c 全量）",
                    "flag-hard": "domain 含 official_flags/official_emblems/official_regional 的 b/c 记录 + "
                                 "stage_e absent_clean/hard_distractor 记录",
                    "ip-brand": "domain 含 luxury_brands/ip_brands 的 b/c 记录 + "
                                "stage_e absent_clean/hard_distractor 记录",
                    "anime-ip": "domain 含 character_ips/subculture_elements 的 b/c 记录 + "
                                "stage_e absent_clean/hard_distractor 记录",
                    "misleading": "stage_e absent_clean/occluded/hard_distractor 全量",
                    "quantity": "stage_c 含 count/relation QA 且 2 个实体的记录",
                    "text-compliance": "stage_b/c 记录（确定性均匀抽取 3,914 条，1 行/图）",
                    "structured_caption": "stage_b/c 全量记录（多视图 template A/B）",
                },
            },
        },
        "notes": [
            "images 全部引用 v2 已有相对路径（validator 以仓库根 --image-root . 解析），未复制/未生成图片。",
            "所有行 variant=6、enable_thinking=False；图片数恒为 1，<image> 仅出现在第一条 user 消息。",
            "多视图：同一张 v2 图可在同一 cate 下生成多条任务视图；同一 (cate, 图片) 内的行以确定性"
            " anchor 保证第一问不同，因此内容不完全重复。",
            "所有新增问答的事实来自 v2 记录字段或 entities_json/bboxes 直接计算；不含编造。",
            "text-compliance 一致/不一致各半；不一致描述至少一项与真值不同，并在答案中写出「正确应为」真值。",
            "structured_caption 的 assistant 为 json.dumps(ensure_ascii=False) 合法 JSON，字段全部来自 v2 记录。",
            "文本 token 估算：中文 chars/1.6 + 其余 chars/4，另加图片×512；全部行 ≤8192。",
            "生成确定性：固定 seed、按 v2_sample_id 排序、按 (seed,cate,图片,视图) 派生随机流。",
        ],
        "examples": examples,
        "output": {
            "parquet": str(out_parquet.relative_to(PROJECT_ROOT)),
            "parquet_bytes": out_parquet.stat().st_size if out_parquet.exists() else None,
            "parquet_sha256": hashlib.sha256(out_parquet.read_bytes()).hexdigest()
            if out_parquet.exists() else None,
            "columns": [field.name for field in SCHEMA],
        },
    }
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build V6 balanced single-image mid-training rows from Datawheel v2")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="v2 dataset.parquet 路径")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="输出目录（默认 output/datawheel_v6_balanced/parts）")
    parser.add_argument("--seed", type=int, default=SEED, help="确定性抽样 seed（默认 42）")
    args = parser.parse_args(argv)

    dataset_path = args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset)
    if not dataset_path.is_file():
        print(f"[error] dataset 不存在：{dataset_path}", file=sys.stderr)
        return 2
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_stats: Dict[str, Any] = {}
    source = load_source(dataset_path, source_stats)
    records_pool = source["records"]
    print(f"[load] v2 rows={source['source_rows']} train={source['train_rows']} "
          f"b/c/e 可用记录={len(records_pool)} "
          f"({source_stats['records_by_stage']}) "
          f"颜色冲突={source_stats['color_conflicts']}")

    build_stats: Dict[str, Any] = {}
    try:
        records, build_stats = build_all(records_pool, args.seed, build_stats)
    except QuotaError as exc:
        print(f"[error] 配额不足，未写任何产物：{exc}", file=sys.stderr)
        return 2

    image_exists: Dict[str, bool] = {}
    for record in records:
        for image in record["images"]:
            image_exists.setdefault(image, (PROJECT_ROOT / image).is_file())
    check = self_check(records, image_exists)

    clean_rows = [{
        "messages": record["messages"],
        "images": record["images"],
        "enable_thinking": False,
        "sample_id": record["sample_id"],
        "cate_id": record["cate_id"],
        "variant": VARIANT,
    } for record in records]
    table = pa.Table.from_pylist(clean_rows, schema=SCHEMA)
    out_parquet = out_dir / OUT_PARQUET
    pq.write_table(table, out_parquet, compression="zstd")

    manifest = build_manifest(records, check, build_stats, source_stats,
                              dataset_path, out_parquet, args.seed)
    out_manifest = out_dir / OUT_MANIFEST
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[single_rows] wrote {out_parquet.relative_to(PROJECT_ROOT)} "
          f"({table.num_rows} rows, {out_parquet.stat().st_size / 1024:.0f} KB)")
    print(f"[single_rows] wrote {out_manifest.relative_to(PROJECT_ROOT)}")
    print("[single_rows] cate_dist:", json.dumps(check["cate_dist"], ensure_ascii=False))
    print("[single_rows] row_type_dist:", dict(manifest["row_type_dist"]))
    print("[single_rows] images: unique=", manifest["images"]["unique_paths"],
          "total_refs=", manifest["images"]["total_references"],
          "reuse_count_dist=", manifest["images"]["reuse_count_dist"])
    print("[single_rows] tokens: ", manifest["text_stats_all"]["tokens_est_incl_image_512"],
          "over_max_len=", manifest["text_stats_all"]["over_max_len"])
    print("[single_rows] exact_duplicate_content_rows=",
          check["exact_duplicate_content_rows"], "build_duplicates=",
          build_stats.get("duplicate_content_accepts", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
