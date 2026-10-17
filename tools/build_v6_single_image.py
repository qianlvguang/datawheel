#!/usr/bin/env python3
"""Datawheel v2 → Qwen3.5-27B 中训练 V6 通道「单图行」构建器。

把 v2 的 stage_b/c/e 单图记录转成 ms-swift 4.3.2 中训练 parquet 行：
  columns = messages / images / enable_thinking / sample_id / cate_id / variant
所有行 variant=6（训练插件在编码时自动映射为 channel='V6' → loss_V6），
enable_thinking=False，images 只引用 v2 已有的相对路径（不复制/不生成图片）。

行类型与配额（默认 seed=42 确定性抽样，精确行数）：
  1) 单图多轮 QA      1200 行，cate_id ∈ {misleading, quantity, apparel,
                        flag-hard, ip-brand, anime-ip}，每行 2~3 条
                        answer_type 互不相同的 QA；<image> 只在第 1 条 user 消息；
  2) 结构化 JSON caption 400 行，cate_id=structured_caption，assistant 为合法 JSON；
  3) 文本一致性判断     250 行，cate_id=text-compliance，125 一致 + 125 故意不一致。

数据卫生：
  - 只取 split=="train" 的 v2 记录，绝不使用 val / test_unseen_*，避免把 v2
    泛化评测集泄入中训练（manifest 记录 split_policy="train_only"）；
  - 同一个 v2 记录（同一张图）在 1850 行里只使用一次（行级 images 不重复）；
  - 颜色标签取自同一实体在 train 记录中的 v2 color QA（资产级调色板属性）。

用法（在 Coding/datawheel/ 下）：
  .venv/bin/python tools/build_v6_single_image.py
  .venv/bin/python tools/build_v6_single_image.py --qa-rows 1200 --caption-rows 400 \
      --compliance-rows 250 --seed 42

产物：
  output/datawheel_v6_pilot/parts/single_image.parquet
  output/datawheel_v6_pilot/parts/single_image_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "output" / "datawheel_v2" / "dataset.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_pilot" / "parts"

VARIANT = 6
SEED = 42
IMAGE_TOKEN_ESTIMATE = 512
MAX_LEN = 8192

STAGE_B = "stage_b_single_element"
STAGE_C = "stage_c_multi_element"
STAGE_E = "stage_e_negative_evidence"
STAGES = (STAGE_B, STAGE_C, STAGE_E)

# 多轮 QA 六类的输出顺序（也是 sample_id 分配顺序的一部分）
CATE_ORDER = ("misleading", "quantity", "apparel", "flag-hard", "ip-brand", "anime-ip")

MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("messages", MESSAGES_SCHEMA),
    ("images", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
    ("sample_id", pa.string()),
    ("cate_id", pa.string()),
    ("variant", pa.int32()),
])

CAPTION_PROMPT = (
    "请以严格 JSON 输出该商品的结构化描述："
    "品类、工艺位置、印花元素（名称/边界框/颜色）、面料与场景。"
)
COMPLIANCE_PROMPT_TMPL = "判断以下描述与图片是否一致，并给出简短依据：{statement}"

# 服装/箱包/鞋履/帽饰底板关键词（Lead 指定：stage_b 中此类底板按 md5 规则归 apparel）
APPAREL_ZH_KEYWORDS = ("上装", "裙装", "连衣裙", "裤装", "下装", "箱包", "鞋履", "帽饰")
APPAREL_EN_KEYWORDS = (
    "top", "t-shirt", "tshirt", "tee", "dress", "pants", "trousers", "jeans",
    "bag", "backpack", "tote", "shoe", "sneaker", "hat", "headwear", "cap",
    "beanie", "shirt", "coat", "jacket", "sweater", "cardigan",
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")

ASPECT_ZH = {
    "bbox": "边界框",
    "color": "颜色",
    "technique": "印花工艺",
    "position": "工艺位置",
    "relation": "相对方位",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> float:
    """按字符数估算文本 token：中文 chars/1.6 + 其余（英文/数字/符号/空白）chars/4。

    与 tools/validate_swift_v6.py 中的公式保持一致（保守估算，仅用于行级长度门禁）。
    """
    if not text:
        return 0.0
    cjk = len(CJK_RE.findall(text))
    return cjk / 1.6 + (len(text) - cjk) / 4.0


def messages_text(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def messages_chars(messages: Sequence[Dict[str, str]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def fmt_bbox(bbox: Sequence[int]) -> str:
    return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"


def md5_hash_mod(sample_id: str, mod: int = 3) -> int:
    """md5(sample_id) 前 4 位十六进制 → int → % mod（Lead 指定的确定性分流）。"""
    digest = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest[:4], 16) % mod


def is_apparel_substrate(row: Dict[str, Any]) -> bool:
    zh = (row.get("substrate_category_zh") or "").lower()
    en = (row.get("substrate_category_en") or "").lower()
    return any(k in zh for k in APPAREL_ZH_KEYWORDS) or any(k in en for k in APPAREL_EN_KEYWORDS)


def classify_cate(cand: Dict[str, Any]) -> Tuple[str, str]:
    """Lead 指定的 cate 映射（按优先级，取第一条命中），返回 (cate_id, 规则命中路径)。

    1) stage_e / task_family 含 negative|insufficient  -> misleading
    2) stage_c 且任一 QA answer_type ∈ {count, relation} -> quantity
    3) stage_b 且底板属于上装/裙装/裤装/箱包/鞋履/帽饰 且 md5(sample_id) 前4位 % 3 == 0 -> apparel
    4) domain 含 official_flags|official_emblems|official_regional -> flag-hard
    5) domain 含 luxury_brands|ip_brands -> ip-brand
    6) domain 含 character_ips|subculture -> anime-ip
    7) 其余 -> apparel
    """
    stage = cand["task_stage"]
    family = (cand.get("task_family") or "").lower()
    answer_types = cand["answer_types"]
    if stage == STAGE_E or "negative" in family or "insufficient" in family:
        return "misleading", "stage_e_or_negative_family"
    if stage == STAGE_C and ({"count", "relation"} & answer_types):
        return "quantity", "stage_c_count_relation"
    if stage == STAGE_B and cand["is_apparel"] and md5_hash_mod(cand["v2_sample_id"]) == 0:
        return "apparel", "stage_b_apparel_md5"
    domain = cand.get("domain") or ""
    if any(k in domain for k in ("official_flags", "official_emblems", "official_regional")):
        return "flag-hard", "domain_official"
    if any(k in domain for k in ("luxury_brands", "ip_brands")):
        return "ip-brand", "domain_brand"
    if any(k in domain for k in ("character_ips", "subculture")):
        return "anime-ip", "domain_ip_character"
    return "apparel", "fallback_apparel"


# ---------------------------------------------------------------------------
# 读取 v2 + 候选池
# ---------------------------------------------------------------------------
def load_candidates(dataset_path: Path, stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    columns = [
        "sample_id", "task_stage", "task_family", "split", "domain",
        "substrate_category_zh", "substrate_category_en", "scene_style_zh",
        "placement_zone", "print_technique_zh", "bboxes_norm_1000",
        "entities_json", "entity_ids_json", "caption_zh", "qa_pairs_json",
        "image_path",
    ]
    table = pq.read_table(dataset_path, columns=columns)
    rows = table.to_pylist()
    stats["source_rows"] = len(rows)
    stats["source_columns"] = columns
    stats["source_split_dist"] = dict(Counter(r["split"] for r in rows))

    cands: List[Dict[str, Any]] = []
    skipped = Counter()
    for row in rows:
        stage = row["task_stage"]
        if stage not in STAGES:
            skipped["stage_not_bce"] += 1
            continue
        if row["split"] != "train":
            skipped["split_not_train"] += 1
            continue
        image_rel = row.get("image_path") or ""
        if not image_rel or not (PROJECT_ROOT / image_rel).is_file():
            skipped["image_missing"] += 1
            continue
        qa = json.loads(row["qa_pairs_json"] or "[]")
        entities = json.loads(row["entities_json"] or "[]")
        entity_ids = json.loads(row["entity_ids_json"] or "[]")
        bboxes = json.loads(row["bboxes_norm_1000"] or "[]")
        # 多轮 QA 行只需要 QA 对；stage_e 的 negative_absent_clean 记录没有实体标注，
        # 但正是「正确否定」的关键样本，因此实体为空不跳过（caption/compliance 另行过滤）。
        if not qa:
            skipped["empty_qa"] += 1
            continue
        if any(not (q.get("question_zh") or "").strip() for q in qa):
            skipped["empty_question"] += 1
            continue
        if any(not (q.get("answer_zh") or "").strip() for q in qa):
            skipped["empty_answer"] += 1
            continue
        answer_types = [q.get("answer_type") or "" for q in qa]
        cand = {
            "v2_sample_id": row["sample_id"],
            "task_stage": stage,
            "task_family": row["task_family"] or "",
            "domain": row["domain"] or "",
            "substrate_category_zh": row["substrate_category_zh"] or "",
            "substrate_category_en": row["substrate_category_en"] or "",
            "scene_style_zh": row["scene_style_zh"] or "",
            "placement_zone": row["placement_zone"] or "",
            "print_technique_zh": row["print_technique_zh"] or "",
            "caption_zh": row["caption_zh"] or "",
            "image_path": image_rel,
            "qa": qa,
            "answer_types": set(answer_types),
            "entities": entities,
            "entity_ids": entity_ids,
            "bboxes": bboxes,
            "has_entities": bool(entities and entity_ids and bboxes),
            "is_apparel": is_apparel_substrate(row),
        }
        if len(cand["answer_types"]) < 2:
            skipped["qa_answer_types_lt2"] += 1
            continue
        cand["cate_id"], cand["cate_rule"] = classify_cate(cand)
        cands.append(cand)

    stats["skipped"] = dict(skipped)
    stats["candidates"] = len(cands)
    stats["candidate_pool_dist"] = dict(Counter(c["cate_id"] for c in cands))
    stats["candidate_stage_dist"] = dict(Counter(c["task_stage"] for c in cands))
    return cands, stats


def build_entity_color_map(cands: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """(domain, stem) → 实体代表色。仅用 train 的 b/c 记录颜色 QA（无 split 泄漏）。"""
    mapping: Dict[Tuple[str, str], Counter] = {}
    for cand in cands:
        if cand["task_stage"] not in (STAGE_B, STAGE_C):
            continue
        colors = [q["answer_zh"] for q in cand["qa"] if q.get("answer_type") == "color"]
        if not colors:
            continue
        if cand["task_stage"] == STAGE_B and len(cand["entity_ids"]) == 1:
            key = (cand["entity_ids"][0]["domain"], cand["entity_ids"][0]["stem"])
        elif cand["task_stage"] == STAGE_C and len(cand["entity_ids"]) >= 2:
            # stage_c 的 color QA 问的是「次要图案」，对应第 2 个实体
            key = (cand["entity_ids"][1]["domain"], cand["entity_ids"][1]["stem"])
        else:
            continue
        mapping.setdefault(key, Counter())[colors[0]] += 1
    # 每个实体保留出现次数最多的颜色（确定性：sorted 后再取 max）
    return {key: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            for key, counter in mapping.items()}


def build_vocab(cands: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    colors, techniques, positions, relations = set(), set(), set(), set()
    for cand in cands:
        techniques.add(cand["print_technique_zh"])
        for q in cand["qa"]:
            ans = q["answer_zh"]
            atype = q.get("answer_type")
            if atype == "color":
                colors.add(ans)
            elif atype == "location":
                positions.add(ans)
            elif atype == "relation":
                relations.add(ans)
    return {
        "colors": sorted(colors),
        "techniques": sorted(techniques),
        "positions": sorted(positions),
        "relations": sorted(relations),
    }


# ---------------------------------------------------------------------------
# 行构造
# ---------------------------------------------------------------------------
def pick_qa_subset(qa: Sequence[Dict[str, Any]], rng: random.Random) -> Optional[List[Dict[str, Any]]]:
    """选 2~3 条 answer_type 互不相同的 QA，保持原记录中的相对顺序。"""
    groups: "OrderedDict[str, List[Tuple[int, Dict[str, Any]]]]" = OrderedDict()
    for idx, item in enumerate(qa):
        atype = item.get("answer_type") or ""
        groups.setdefault(atype, []).append((idx, item))
    types = list(groups)
    if len(types) < 2:
        return None
    k = 3 if len(types) >= 3 and rng.random() < 0.5 else 2
    k = min(k, len(types))
    chosen_types = rng.sample(types, k) if k < len(types) else list(types)
    chosen = [rng.choice(groups[t]) for t in chosen_types]
    chosen.sort(key=lambda pair: pair[0])
    return [item for _, item in chosen]


def make_multi_turn_row(cand: Dict[str, Any], qa_subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    for i, qa in enumerate(qa_subset):
        content = qa["question_zh"].strip()
        if i == 0:
            content = "<image>\n" + content
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": qa["answer_zh"].strip()})
    return {
        "row_type": "multi_turn_qa",
        "cate_id": cand["cate_id"],
        "cate_rule": cand["cate_rule"],
        "v2_sample_id": cand["v2_sample_id"],
        "v2_stage": cand["task_stage"],
        "messages": messages,
        "images": [cand["image_path"]],
        "qa_rounds": len(qa_subset),
    }


def element_payload(cand: Dict[str, Any], color_map: Dict[Tuple[str, str], str]) -> List[Dict[str, Any]]:
    elements = []
    for entity, eid in zip(cand["entities"], cand["entity_ids"]):
        color = color_map.get((eid["domain"], eid["stem"]))
        item: Dict[str, Any] = {
            "name_zh": entity.get("full_title_zh") or entity.get("entity_name_zh") or "",
            "bbox_1000": [int(v) for v in (entity.get("bbox_1000") or [])],
        }
        if color:
            item["color_zh"] = color
        elements.append(item)
    return elements


def make_caption_row(cand: Dict[str, Any], color_map: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    payload = {
        "category_zh": cand["substrate_category_zh"],
        "scene_style_zh": cand["scene_style_zh"],
        "placement_zone": cand["placement_zone"],
        "print_technique_zh": cand["print_technique_zh"],
        "elements": element_payload(cand, color_map),
        "caption_zh": cand["caption_zh"],
    }
    assistant = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
    return {
        "row_type": "structured_caption",
        "cate_id": "structured_caption",
        "cate_rule": "row_type_fixed",
        "v2_sample_id": cand["v2_sample_id"],
        "v2_stage": cand["task_stage"],
        "messages": [
            {"role": "user", "content": "<image>\n" + CAPTION_PROMPT},
            {"role": "assistant", "content": assistant},
        ],
        "images": [cand["image_path"]],
    }


def fact_bundle(cand: Dict[str, Any], color_map: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
    main_entity = cand["entities"][0]
    main_eid = cand["entity_ids"][0]
    location = next((q["answer_zh"] for q in cand["qa"] if q.get("answer_type") == "location"), None)
    relation = next((q["answer_zh"] for q in cand["qa"] if q.get("answer_type") == "relation"), None)
    return {
        "category": cand["substrate_category_zh"],
        "technique": cand["print_technique_zh"],
        "main_name": main_entity.get("full_title_zh") or main_entity.get("entity_name_zh") or "",
        "main_bbox": [int(v) for v in (main_entity.get("bbox_1000") or cand["bboxes"][0])],
        "main_color": color_map.get((main_eid["domain"], main_eid["stem"])),
        "position": location,
        "relation": relation,
    }


def make_consistent_statement(facts: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
    options: List[Tuple[str, str, str]] = []
    if facts["main_color"]:
        options.append((
            "color",
            f"主视觉图案【{facts['main_name']}】的主体颜色为{facts['main_color']}。",
            f"该图案代表色取自资产调色板，为{facts['main_color']}。",
        ))
    options.append((
        "bbox",
        f"主视觉图案【{facts['main_name']}】的归一化边界框为 {fmt_bbox(facts['main_bbox'])}。",
        "边界框由形变后的可见图案区域反算，与描述一致。",
    ))
    options.append((
        "technique",
        f"图中商品为{facts['category']}，印花采用{facts['technique']}工艺。",
        f"该商品底板为{facts['category']}，印花工艺记录为{facts['technique']}。",
    ))
    if facts["position"]:
        options.append((
            "position",
            f"主视觉图案位于{facts['position']}。",
            f"该图案的工艺位置记录为{facts['position']}。",
        ))
    if facts["relation"]:
        options.append((
            "relation",
            f"主视觉图案与次要图案的相对位置关系为{facts['relation']}。",
            f"两图元中心点比较得到的相对方位为{facts['relation']}。",
        ))
    aspect, statement, evidence = rng.choice(options)
    answer = f"一致。依据：{evidence}"
    return statement, answer


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
    return [max(0, ymin - 100), max(0, xmin - 100), max(0, ymax - 100), max(0, xmax - 100)]


def pick_other(values: Sequence[str], true_value: str, rng: random.Random) -> Optional[str]:
    pool = [v for v in values if v and v != true_value]
    return rng.choice(pool) if pool else None


def make_inconsistent_statement(
    facts: Dict[str, Any],
    vocab: Dict[str, List[str]],
    rng: random.Random,
) -> Tuple[Optional[str], Optional[str]]:
    """构造一条与记录事实确实不同的描述，返回 (statement, assistant_answer)。"""
    aspects: List[str] = ["bbox", "technique"]
    if facts["main_color"]:
        aspects.append("color")
    if facts["position"]:
        aspects.append("position")
    if facts["relation"]:
        aspects.append("relation")
    rng.shuffle(aspects)
    for aspect in aspects:
        if aspect == "bbox":
            fake = shift_bbox(facts["main_bbox"], rng)
            true_value = fmt_bbox(facts["main_bbox"])
            statement = f"主视觉图案【{facts['main_name']}】的归一化边界框为 {fmt_bbox(fake)}。"
            evidence = "边界框由形变后的可见图案区域反算。"
        elif aspect == "technique":
            fake = pick_other(vocab["techniques"], facts["technique"], rng)
            if fake is None:
                continue
            true_value = facts["technique"]
            statement = f"图中商品为{facts['category']}，印花采用{fake}工艺。"
            evidence = f"该商品底板为{facts['category']}，印花工艺记录为{true_value}。"
        elif aspect == "color":
            fake = pick_other(vocab["colors"], facts["main_color"], rng)
            if fake is None:
                continue
            true_value = facts["main_color"]
            statement = f"主视觉图案【{facts['main_name']}】的主体颜色为{fake}。"
            evidence = f"该图案代表色取自资产调色板，为{true_value}。"
        elif aspect == "position":
            fake = pick_other(vocab["positions"], facts["position"], rng)
            if fake is None:
                continue
            true_value = facts["position"]
            statement = f"主视觉图案位于{fake}。"
            evidence = f"该图案的实际位置为{true_value}。"
        else:  # relation
            fake = pick_other(vocab["relations"], facts["relation"], rng)
            if fake is None:
                continue
            true_value = facts["relation"]
            statement = (f"主视觉图案与次要图案的相对位置关系为{fake}。")
            evidence = f"两图元中心点比较得到的相对方位为{true_value}。"
        answer = (
            f"不一致：描述中的{ASPECT_ZH[aspect]}与图片不符"
            f"（正确应为{true_value}）。依据：{evidence}"
        )
        if statement and true_value:
            return statement, answer
    return None, None


def make_compliance_row(
    cand: Dict[str, Any],
    color_map: Dict[Tuple[str, str], str],
    vocab: Dict[str, List[str]],
    consistent: bool,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    facts = fact_bundle(cand, color_map)
    if consistent:
        statement, answer = make_consistent_statement(facts, rng)
    else:
        statement, answer = make_inconsistent_statement(facts, vocab, rng)
        if statement is None:
            return None
    messages = [
        {"role": "user", "content": "<image>\n" + COMPLIANCE_PROMPT_TMPL.format(statement=statement)},
        {"role": "assistant", "content": answer},
    ]
    return {
        "row_type": "text_compliance",
        "cate_id": "text-compliance",
        "cate_rule": "consistent" if consistent else "inconsistent",
        "v2_sample_id": cand["v2_sample_id"],
        "v2_stage": cand["task_stage"],
        "compliance_label": "consistent" if consistent else "inconsistent",
        "messages": messages,
        "images": [cand["image_path"]],
    }


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def take_rows(
    pool: Sequence[Dict[str, Any]],
    n: int,
    rng: random.Random,
    used: set,
) -> List[Dict[str, Any]]:
    avail = sorted((c for c in pool if c["v2_sample_id"] not in used), key=lambda c: c["v2_sample_id"])
    rng.shuffle(avail)
    return avail[:n]


def split_quota(total: int, parts: int) -> List[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def build_records(
    dataset_path: Path,
    seed: int,
    qa_rows: int,
    caption_rows: int,
    compliance_rows: int,
    log: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    cands, stats = load_candidates(dataset_path, {})
    color_map = build_entity_color_map(cands)
    vocab = build_vocab(cands)

    rng = random.Random(seed)
    used: set = set()
    records: List[Dict[str, Any]] = []

    # ---- 1) 单图多轮 QA：六类等额（默认每类 200 行） ----
    by_cate: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATE_ORDER}
    for cand in cands:
        by_cate[cand["cate_id"]].append(cand)
    quotas = dict(zip(CATE_ORDER, split_quota(qa_rows, len(CATE_ORDER))))
    qa_actual = Counter()
    for cate in CATE_ORDER:
        target = quotas[cate]
        pool = by_cate[cate]
        picked = take_rows(pool, target, rng, used)
        if len(picked) < target:
            print(f"[warn] cate={cate} 候选池不足：需要 {target}，只有 {len(picked)}", file=sys.stderr)
        for cand in picked:
            qa_subset = pick_qa_subset(cand["qa"], rng)
            if qa_subset is None:
                continue
            records.append(make_multi_turn_row(cand, qa_subset))
            used.add(cand["v2_sample_id"])
            qa_actual[cate] += 1
    qa_rows_actual = sum(qa_actual.values())

    # ---- 2) 结构化 JSON caption：stage_b / stage_c 各半 ----
    caption_actual = Counter()
    if caption_rows > 0:
        b_quota, c_quota = caption_rows // 2, caption_rows - caption_rows // 2
        b_pool = [c for c in cands if c["task_stage"] == STAGE_B and c["has_entities"]]
        c_pool = [c for c in cands if c["task_stage"] == STAGE_C and c["has_entities"]]
        b_picked = take_rows(b_pool, b_quota, rng, used)
        for cand in b_picked:
            used.add(cand["v2_sample_id"])
        c_picked = take_rows(c_pool, c_quota, rng, used)
        for cand in c_picked:
            used.add(cand["v2_sample_id"])
        for i in range(max(len(b_picked), len(c_picked))):
            for picked, stage_key in ((b_picked, "stage_b"), (c_picked, "stage_c")):
                if i < len(picked):
                    records.append(make_caption_row(picked[i], color_map))
                    caption_actual[stage_key] += 1

    # ---- 3) 文本一致性判断：stage_b / stage_c 约 60/40，一致/不一致各半 ----
    compliance_actual = Counter()
    if compliance_rows > 0:
        n_b = int(round(compliance_rows * 0.6))
        n_c = compliance_rows - n_b
        b_pool = [c for c in cands if c["task_stage"] == STAGE_B and c["has_entities"]]
        c_pool = [c for c in cands if c["task_stage"] == STAGE_C and c["has_entities"]]
        b_picked = take_rows(b_pool, n_b, rng, used)
        for cand in b_picked:
            used.add(cand["v2_sample_id"])
        c_picked = take_rows(c_pool, n_c, rng, used)
        for cand in c_picked:
            used.add(cand["v2_sample_id"])
        ordered: List[Dict[str, Any]] = []
        for i in range(max(len(b_picked), len(c_picked))):
            if i < len(b_picked):
                ordered.append(b_picked[i])
            if i < len(c_picked):
                ordered.append(c_picked[i])
        for cand in ordered:
            # 严格一半一致、一半故意不一致（按已落入行数决定，跳过异常行也不破坏配额）
            consistent = compliance_actual["consistent"] == compliance_actual["inconsistent"]
            row = make_compliance_row(cand, color_map, vocab, consistent, rng)
            if row is None:  # 理论上不会发生（每类都有可篡改维度）
                continue
            records.append(row)
            compliance_actual[row["compliance_label"]] += 1

    # ---- 分配 sample_id（V6S_{:06d}）----
    for i, record in enumerate(records):
        record["sample_id"] = f"V6S_{i + 1:06d}"

    log.update({
        "elapsed_build_seconds": round(time.time() - t0, 3),
        "color_map_entities": len(color_map),
        "vocab": {k: len(v) for k, v in vocab.items()},
        "qa_quota_target": quotas,
        "qa_actual": dict(qa_actual),
        "qa_rows_actual": qa_rows_actual,
        "caption_actual": dict(caption_actual),
        "compliance_actual": dict(compliance_actual),
        "used_v2_records": len(used),
    })
    stats["color_map_entities"] = len(color_map)
    return records, stats


# ---------------------------------------------------------------------------
# 自检 / 统计 / 输出
# ---------------------------------------------------------------------------
def self_check(records: Sequence[Dict[str, Any]]) -> None:
    """落盘前的轻量自检：角色交替、占位符数量、长度上限、行键唯一。"""
    seen_ids, seen_keys = set(), set()
    for record in records:
        messages = record["messages"]
        images = record["images"]
        sid = record["sample_id"]
        if sid in seen_ids:
            raise AssertionError(f"sample_id 重复：{sid}")
        seen_ids.add(sid)
        key = (sid, record["cate_id"], tuple(images))
        if key in seen_keys:
            raise AssertionError(f"行级键重复：{key}")
        seen_keys.add(key)
        if not messages or len(messages) % 2 != 0:
            raise AssertionError(f"{sid}: messages 非 user/assistant 交替结构")
        for i, message in enumerate(messages):
            expected = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise AssertionError(f"{sid}: 第 {i} 条消息角色应为 {expected}")
            if not str(message["content"]).strip():
                raise AssertionError(f"{sid}: 空 content")
        if len(images) != 1:
            raise AssertionError(f"{sid}: 单图行 images 长度应为 1")
        placeholder_total = sum(str(m["content"]).count("<image>") for m in messages)
        placeholder_first = str(messages[0]["content"]).count("<image>")
        if placeholder_total != len(images) or placeholder_first != len(images):
            raise AssertionError(f"{sid}: <image> 数量与 images 不一致")
        text = messages_text(messages)
        est = estimate_text_tokens(text) + len(images) * IMAGE_TOKEN_ESTIMATE
        if est > MAX_LEN:
            raise AssertionError(f"{sid}: 估算长度 {est:.1f} 超过 MAX_LEN={MAX_LEN}")
    expected = 1850
    if len(records) != expected:
        print(f"[warn] 行数 {len(records)} != 期望 {expected}（配额参数可能被修改）", file=sys.stderr)


def stats_summary(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "mean": round(statistics.fmean(values), 2),
        "max": round(max(values), 2),
    }


def truncate(text: str, limit: int = 700) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def build_manifest(
    records: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
    log: Dict[str, Any],
    dataset_path: Path,
    out_parquet: Path,
    seed: int,
    quotas: Dict[str, int],
) -> Dict[str, Any]:
    row_type_dist = Counter(r["row_type"] for r in records)
    cate_dist = Counter(r["cate_id"] for r in records)
    cate_rule_dist: Dict[str, Dict[str, int]] = {}
    for record in records:
        if record["row_type"] == "multi_turn_qa":
            cate_rule_dist.setdefault(record["cate_id"], Counter())[record["cate_rule"]] += 1
    stage_dist: Dict[str, Dict[str, int]] = {}
    for record in records:
        stage_dist.setdefault(record["v2_stage"], Counter())[record["row_type"]] += 1
    image_count_dist = Counter(len(r["images"]) for r in records)
    all_images = [img for r in records for img in r["images"]]
    unique_images = len(set(all_images))

    per_type_text: Dict[str, Dict[str, Any]] = {}
    for row_type in ("multi_turn_qa", "structured_caption", "text_compliance"):
        subset = [r for r in records if r["row_type"] == row_type]
        chars = [messages_chars(r["messages"]) for r in subset]
        tokens = [estimate_text_tokens(messages_text(r["messages"])) for r in subset]
        total_tokens = [tok + len(r["images"]) * IMAGE_TOKEN_ESTIMATE for tok, r in zip(tokens, subset)]
        over = sum(1 for v in total_tokens if v > MAX_LEN)
        per_type_text[row_type] = {
            "rows": len(subset),
            "text_chars": stats_summary(chars),
            "text_tokens_est": stats_summary(tokens),
            "tokens_est_incl_image_512": stats_summary(total_tokens),
            "over_max_len": over,
        }
    all_chars = [messages_chars(r["messages"]) for r in records]
    all_text_tokens = [estimate_text_tokens(messages_text(r["messages"])) for r in records]
    all_total_tokens = [tok + len(r["images"]) * IMAGE_TOKEN_ESTIMATE
                        for tok, r in zip(all_text_tokens, records)]
    qa_round_dist = Counter(r["qa_rounds"] for r in records if r["row_type"] == "multi_turn_qa")
    compliance_label_dist = Counter(r["compliance_label"] for r in records
                                    if r["row_type"] == "text_compliance")

    # 每类抽 1 条示例（manifest 只存截断后的可读版本）
    examples = []
    for row_type in ("multi_turn_qa", "structured_caption", "text_compliance"):
        record = next((r for r in records if r["row_type"] == row_type), None)
        if record is None:
            continue
        examples.append({
            "row_type": row_type,
            "sample_id": record["sample_id"],
            "v2_sample_id": record["v2_sample_id"],
            "cate_id": record["cate_id"],
            "images": record["images"],
            "messages": [{"role": m["role"], "content": truncate(m["content"])}
                         for m in record["messages"]],
        })

    manifest = {
        "part": "single_image",
        "rows": len(records),
        "variant": VARIANT,
        "seed": seed,
        "split_policy": "train_only",
        "split_dist": dict(Counter("train" for _ in records)),
        "row_type_dist": dict(row_type_dist),
        "row_type_quota_target": {
            "multi_turn_qa": quotas["qa"],
            "structured_caption": quotas["caption"],
            "text_compliance": quotas["compliance"],
        },
        "cate_dist": dict(cate_dist),
        "cate_rule_dist": {cate: dict(counter) for cate, counter in cate_rule_dist.items()},
        "v2_stage_crosstab": {stage: dict(counter) for stage, counter in stage_dist.items()},
        "compliance_label_dist": dict(compliance_label_dist),
        "qa_round_dist": {str(k): v for k, v in sorted(qa_round_dist.items())},
        "image_count_dist": {str(k): v for k, v in sorted(image_count_dist.items())},
        "images": {
            "total_references": len(all_images),
            "unique_paths": unique_images,
            "reused_paths": len(all_images) - unique_images,
            "all_files_exist": all((PROJECT_ROOT / img).is_file() for img in set(all_images)),
        },
        "unique_v2_records_used": len({r["v2_sample_id"] for r in records}),
        "uniqueness": {
            "sample_id_unique": len({r["sample_id"] for r in records}) == len(records),
            "row_key_sample_id_cate_images_unique": len({(r["sample_id"], r["cate_id"], tuple(r["images"]))
                                                         for r in records}) == len(records),
        },
        "text_stats_per_row_type": per_type_text,
        "text_stats_all": {
            "rows": len(records),
            "text_chars": stats_summary(all_chars),
            "text_tokens_est": stats_summary(all_text_tokens),
            "text_tokens_est_total": round(sum(all_text_tokens), 1),
            "tokens_est_incl_image_512": stats_summary(all_total_tokens),
            "max_len": MAX_LEN,
            "over_max_len": sum(1 for v in all_total_tokens if v > MAX_LEN),
        },
        "candidate_pool_dist": stats.get("candidate_pool_dist", {}),
        "candidate_stage_dist": stats.get("candidate_stage_dist", {}),
        "source": {
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "rows": stats.get("source_rows"),
            "sha256": stats.get("source_sha256"),
            "filters": {
                "task_stage": list(STAGES),
                "split": ["train"],
                "image_exists": True,
                "min_distinct_answer_types": 2,
            },
            "skipped": stats.get("skipped", {}),
        },
        "color_source": "v2 train b/c color QA（实体资产主色调，仅 train 汇总）",
        "cate_mapping_rule": (
            "1) stage_e 或 task_family 含 negative/insufficient -> misleading; "
            "2) stage_c 且任一 QA answer_type in {count, relation} -> quantity; "
            "3) stage_b 且底板为服装/箱包/鞋履/帽饰 且 md5(sample_id)[:4]%3==0 -> apparel; "
            "4) domain 含 official_flags/official_emblems/official_regional -> flag-hard; "
            "5) domain 含 luxury_brands/ip_brands -> ip-brand; "
            "6) domain 含 character_ips/subculture -> anime-ip; 7) 其余 -> apparel。"
        ),
        "notes": [
            "images 均为仓库相对路径（validator 以仓库根 --image-root . 解析），未复制/未生成图片。",
            "所有行 variant=6，训练插件自动映射 channel='V6' -> loss_V6；enable_thinking=False。",
            "同一条 v2 记录（同一张图）只使用一次；1850 行对应 1850 个不同 v2 记录/图片。",
            "文本 token 估算公式：中文 chars/1.6 + 其余 chars/4；另计 images*512 个视觉 token。",
            "stage_e 的 misleading 行含 2~3 轮 QA；assistant 为简洁答案。",
        ],
        "output": {
            "parquet": str(out_parquet.relative_to(PROJECT_ROOT)),
            "parquet_bytes": out_parquet.stat().st_size if out_parquet.exists() else None,
            "columns": [field.name for field in SCHEMA],
        },
        "examples": examples,
    }
    return manifest


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build V6 single-image mid-training rows from Datawheel v2")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="v2 dataset.parquet 路径")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="输出目录（默认 output/datawheel_v6_pilot/parts）")
    parser.add_argument("--seed", type=int, default=SEED, help="确定性抽样 seed")
    parser.add_argument("--qa-rows", type=int, default=1200, help="单图多轮 QA 行数")
    parser.add_argument("--caption-rows", type=int, default=400, help="结构化 JSON caption 行数")
    parser.add_argument("--compliance-rows", type=int, default=250, help="文本一致性判断行数")
    args = parser.parse_args(argv)

    dataset_path = args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset)
    if not dataset_path.is_file():
        print(f"[error] dataset 不存在：{dataset_path}", file=sys.stderr)
        return 2
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {}
    records, stats = build_records(
        dataset_path=dataset_path,
        seed=args.seed,
        qa_rows=args.qa_rows,
        caption_rows=args.caption_rows,
        compliance_rows=args.compliance_rows,
        log=log,
    )
    self_check(records)

    clean_rows = [{
        "messages": record["messages"],
        "images": record["images"],
        "enable_thinking": False,
        "sample_id": record["sample_id"],
        "cate_id": record["cate_id"],
        "variant": VARIANT,
    } for record in records]
    table = pa.Table.from_pylist(clean_rows, schema=SCHEMA)

    out_parquet = out_dir / "single_image.parquet"
    pq.write_table(table, out_parquet, compression="zstd")

    stats["source_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    quotas = {"qa": args.qa_rows, "caption": args.caption_rows, "compliance": args.compliance_rows}
    manifest = build_manifest(records, stats, log, dataset_path, out_parquet, args.seed, quotas)
    manifest["build_log"] = log
    out_manifest = out_dir / "single_image_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[single_image] wrote {out_parquet.relative_to(PROJECT_ROOT)} "
          f"({table.num_rows} rows, {out_parquet.stat().st_size / 1024:.0f} KB)")
    print(f"[single_image] wrote {out_manifest.relative_to(PROJECT_ROOT)}")
    print("[single_image] row_type_dist:", dict(manifest["row_type_dist"]))
    print("[single_image] cate_dist:", dict(manifest["cate_dist"]))
    print("[single_image] qa_round_dist:", manifest["qa_round_dist"],
          "compliance:", manifest["compliance_label_dist"])
    print("[single_image] text_tokens_est:", manifest["text_stats_all"]["text_tokens_est"],
          "over_max_len:", manifest["text_stats_all"]["over_max_len"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
