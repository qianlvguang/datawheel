#!/usr/bin/env python3
"""Datawheel v2 → V6 balanced 多图参考匹配行（reference_match，6,707 行）。

把 v2 **train split** 的 B/C/D/E 商品图扩成「多图 in-context 参考匹配」训练行，
完全对齐 Qwen3.5-27B 中训练 V6 的 ms-swift 数据格式：
  columns = messages / images / enable_thinking / sample_id / cate_id / variant

每行结构（与 `tools/build_v6_reference_rows.py` 试点版同构、规模/视图更多）：
- 第 1 张图：商品实拍图
  * stage_b / stage_c / stage_e：直接引用 v2 现有 640x640 商品图；
  * stage_d：裁掉顶部 200px 候选参考带，得到 640x640 商品图并保存到 images/query/，
    bbox 同步从 640x840 映射到裁后坐标（只取 train split）；
- 第 2~4 张图：2~3 张印刷图元的标准参考渲染图（白底 384x384 JPEG），
  按 (domain, stem) 复用缓存；负例只用 **train 可见实体** 的 dHash 视觉近邻（困难负例），
  避免把 v2 泛化评测集的 held-out 实体参考图泄入训练；
- 匹配标签：约 70% match（候选含商品上的印刷实体 + 1~2 个困难负例），
  约 30% no_match（候选全部为负例，不含印刷实体）；
- assistant 回答「与候选X一致」/「没有候选参考图与商品图案一致；商品上的图案是【…】」，
  并给出商品图案 bbox；50% 行追加第二轮 bbox 问答；
- 多视图：同一张商品图允许生成 1~3 行（不同印刷目标 / 候选集 / 问法）。
  C 阶段一张商品图有 2 处印刷图案，用「两处图案中偏上/偏下/偏左/偏右的那处」消歧，
  每行答案与其新候选集严格一致（不沿用试点图的候选带关系）；
- variant=6（训练插件自动映射为 loss_V6），enable_thinking=False。
- 只取 v2 train split；每行 ≤8192（文本 token 估算 + 512/图）。

产物：
  output/datawheel_v6_balanced/parts/reference_rows.parquet
  output/datawheel_v6_balanced/parts/reference_manifest.json
  output/datawheel_v6_balanced/images/query/*.jpg    # 仅 stage_d 裁出的商品图
  output/datawheel_v6_balanced/images/refs/*.jpg     # 白底候选参考图（按资产缓存）

用法：
  .venv/bin/python tools/build_v6_balanced_reference.py            # 默认 6,707 行
  .venv/bin/python tools/build_v6_balanced_reference.py --rows 60 --out-dir /tmp/v6b_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_datawheel_v2 import DatawheelV2Generator, _md5_bucket, _seed_for  # noqa: E402

V2_DATASET = PROJECT_ROOT / "output" / "datawheel_v2" / "dataset.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_balanced"
DEFAULT_ROWS = 6707
DEFAULT_SEED = 42
MATCH_RATIO = 0.70
VARIANT = 6

CANVAS = 640
D_BAND = 200
D_HEIGHT = 840
REF_SIZE = 384
IMG_EST_TOKENS = 512       # 与 tools/validate_swift_v6.py 默认口径一致
MAX_LEN = 8192

# 只取这些阶段（train split）的商品图
SOURCE_STAGES = (
    "stage_b_single_element",
    "stage_c_multi_element",
    "stage_d_reference_match",
    "stage_e_negative_evidence",
)
STAGE_KIND = {
    "stage_b_single_element": "b",
    "stage_c_multi_element": "c",
    "stage_d_reference_match": "d",
    "stage_e_negative_evidence": "e",
}

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")

MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("messages", MESSAGES_SCHEMA),
    ("images", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
    ("sample_id", pa.string()),
    ("cate_id", pa.string()),
    ("variant", pa.int32()),
])

# 第一轮指令模板：{k}=候选数量，{last}=最后一张图序号，{letters}=候选字母，
# {td}=目标图案描述，{bbox_clause}=单轮样本的 bbox 要求（两轮样本留空，bbox 放到第二轮追问）
INSTRUCTION_TEMPLATES = [
    ("第一张为商品实拍图，后续 {k} 张为候选参考图（{letters}）。请判断{td}与哪一张候选参考图一致；"
     "若均不一致，请明确说明并指出商品上实际图案的内容。{bbox_clause}"),
    ("第 1 张为商品实拍图，第 2~{last} 张为候选参考图（{letters}）。请逐一比对{td}，"
     "回答匹配的候选字母（若无匹配请说明）。{bbox_clause}"),
    ("先看商品实拍图，再看后续 {k} 张候选参考图（{letters}）。{td}与哪一张参考图一致？"
     "若都不一致，请说明并给出商品上实际图案的名称。{bbox_clause}"),
]
ONE_TURN_BBOX_CLAUSE = "同时请给出该图案在商品图中的归一化边界框 [ymin,xmin,ymax,xmax]。"

# 第二轮 bbox 追问模板
BBOX_QUESTION_TEMPLATES = [
    "请给出商品上该图案的归一化边界框 [ymin,xmin,ymax,xmax]。",
    "该图案在商品图中的归一化边界框是多少？请输出 [ymin,xmin,ymax,xmax]。",
    "请定位商品上该图案，输出归一化边界框 [ymin,xmin,ymax,xmax]。",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> float:
    """中文 chars/1.6 + 其余 chars/4（与 validate_swift_v6.py 完全一致）。"""
    if not text:
        return 0.0
    cjk = len(CJK_RE.findall(text))
    return cjk / 1.6 + (len(text) - cjk) / 4.0


def _json_bbox(box: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in box) + "]"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Sequence[float], digits: int = 4) -> Dict[str, Any]:
    values = sorted(float(v) for v in values)
    if not values:
        return {"count": 0, "min": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    n = len(values)

    def pick(q: float) -> float:
        return round(values[min(n - 1, max(0, int(round(q * (n - 1)))))], digits)

    return {
        "count": n,
        "min": round(values[0], digits),
        "p10": pick(0.10),
        "median": pick(0.50),
        "mean": round(sum(values) / n, digits),
        "max": round(values[-1], digits),
    }


def map_bbox_to_crop(box: Sequence[int]) -> List[int]:
    """v2 640x840 全图 [ymin,xmin,ymax,xmax] → 裁掉顶部 200px 后 640x640 商品图坐标。"""
    ymin, xmin, ymax, xmax = [int(v) for v in box]
    ny_min = int(round(((ymin * D_HEIGHT / 1000.0) - D_BAND) / CANVAS * 1000))
    ny_max = int(round(((ymax * D_HEIGHT / 1000.0) - D_BAND) / CANVAS * 1000))
    return [
        max(0, min(1000, ny_min)),
        max(0, min(1000, xmin)),
        max(0, min(1000, ny_max)),
        max(0, min(1000, xmax)),
    ]


def bbox_area_ratio(box: Sequence[int]) -> float:
    return (int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1])) / 1_000_000.0


def bbox_valid(box: Sequence[int]) -> bool:
    if len(box) != 4:
        return False
    y0, x0, y1, x1 = [int(v) for v in box]
    return 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000 and bbox_area_ratio(box) >= 0.0008


def _parse_entities(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("entity_ids_json") or "[]"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [e for e in parsed if isinstance(e, dict) and e.get("domain") and e.get("stem")]


def _parse_boxes(record: Dict[str, Any]) -> List[List[int]]:
    raw = record.get("bboxes_norm_1000") or "[]"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [[int(v) for v in box] for box in parsed if isinstance(box, (list, tuple)) and len(box) == 4]


def _parse_entities_json(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("entities_json") or "[]"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [e for e in parsed if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# 参考图渲染 / 负例选择
# ---------------------------------------------------------------------------
def render_reference(gen: DatawheelV2Generator, domain: str, stem: str, cache: Dict[str, str]) -> Optional[str]:
    """把图元渲染成白底 384x384 JPEG 参考图，返回项目相对路径（按 (domain, stem) 缓存）。"""
    key = f"{domain}__{stem}"
    if key in cache:
        return cache[key]
    rgba = gen._load_foreground(domain, stem, target_w=REF_SIZE * 2)
    if rgba is None:
        return None
    canvas = Image.new("RGB", (REF_SIZE, REF_SIZE), (255, 255, 255))
    ref = rgba.copy()
    ref.thumbnail((REF_SIZE - 24, REF_SIZE - 24), Image.Resampling.LANCZOS)
    x = (REF_SIZE - ref.width) // 2
    y = (REF_SIZE - ref.height) // 2
    white = Image.new("RGBA", ref.size, (255, 255, 255, 255))
    white.alpha_composite(ref)
    canvas.paste(white.convert("RGB"), (x, y))
    out_path = REF_DIR / f"{key}.jpg"
    canvas.save(out_path, format="JPEG", quality=92)
    cache[key] = str(out_path.relative_to(PROJECT_ROOT))
    return cache[key]


def build_asset_tables(gen: DatawheelV2Generator
                       ) -> Tuple[Dict[Tuple[str, str], str],
                                  Dict[Tuple[str, str], int],
                                  Dict[Tuple[str, str], str],
                                  List[Dict[str, str]]]:
    """返回 (train 可见实体参考图路径, dHash 整数指纹, 渲染图 md5, 渲染失败列表)。"""
    ref_map: Dict[Tuple[str, str], str] = {}
    dhash_map: Dict[Tuple[str, str], int] = {}
    render_md5: Dict[Tuple[str, str], str] = {}
    cache: Dict[str, str] = {}
    failures: List[Dict[str, str]] = []
    for domain in sorted(gen.asset_domains):
        for stem in sorted(gen.asset_domains[domain]):
            if _md5_bucket(f"{domain}/{stem}") >= 80:  # 仅 train 可见实体，防 held-out 泄入
                continue
            dhash_hex = gen.asset_domains[domain][stem].get("dhash")
            if isinstance(dhash_hex, str) and dhash_hex:
                try:
                    dhash_map[(domain, stem)] = int(dhash_hex, 16)
                except ValueError:
                    pass
            rel = render_reference(gen, domain, stem, cache)
            if rel is None:
                failures.append({"domain": domain, "stem": stem})
                continue
            ref_map[(domain, stem)] = rel
            render_md5[(domain, stem)] = _sha256_file(PROJECT_ROOT / rel)
    if failures:
        print(f"[reference] WARN: {len(failures)} train-visible assets failed to render, excluded", file=sys.stderr)
    print(f"[reference] train-visible reference pool: {len(ref_map)} rendered, "
          f"{len(dhash_map)} dHash fingerprints")
    return ref_map, dhash_map, render_md5, failures


def dhash_distance(dhash_map: Dict[Tuple[str, str], int], a: Tuple[str, str], b: Tuple[str, str]) -> Optional[int]:
    da, db = dhash_map.get(a), dhash_map.get(b)
    if da is None or db is None:
        return None
    return bin(da ^ db).count("1")


def pick_hard_negatives(gen: DatawheelV2Generator,
                        domain: str,
                        stem: str,
                        k: int,
                        rng: random.Random,
                        excluded: set,
                        ref_map: Dict[Tuple[str, str], str],
                        render_md5: Dict[Tuple[str, str], str],
                        forbidden_hashes: set) -> List[Dict[str, Any]]:
    """同域 dHash 视觉近邻（困难负例），只允许 train 可见实体。

    与 DatawheelV2Generator._hard_negatives 同源，但增加三项约束：
      * 候选必须是 train 可见实体（_md5_bucket < 80），避免 held-out 参考图泄入；
      * excluded 中的实体（商品上其它已印图案 / 已选候选）不重复使用；
      * 渲染图与商品上已印图案完全相同（如法国海外旗与法国三色旗）的资产排除，
        否则 match 候选字母 / no_match 标签会出现歧义。
    """
    neighbors = gen.neighbors.get(f"{domain}/{stem}", [])
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for item in neighbors:
        other = item.get("stem")
        if not other or other in seen or other == stem or other in excluded:
            continue
        key = (domain, other)
        if key not in ref_map:
            continue
        if render_md5.get(key) in forbidden_hashes:
            continue
        seen.add(other)
        candidates.append({"domain": domain, "stem": other,
                           "distance": dhash_distance(dhash_map_global, (domain, stem), key)})
    if len(candidates) < k:
        # 近邻表不足时用同域 train 可见实体补齐（仍保持同域 + 非目标/非禁用）
        pool = [other for other in gen.asset_domains[domain]
                if other != stem and other not in excluded and other not in seen
                and (domain, other) in ref_map
                and render_md5.get((domain, other)) not in forbidden_hashes]
        rng.shuffle(pool)
        for other in pool[: k - len(candidates)]:
            candidates.append({"domain": domain, "stem": other, "distance": None})
    chosen = rng.sample(candidates, min(k, len(candidates)))
    return chosen


# dHash 指纹表（main 中填充，供 pick_hard_negatives 使用；避免层层传递）
dhash_map_global: Dict[Tuple[str, str], int] = {}


# ---------------------------------------------------------------------------
# 视图构建：从 v2 train 记录得到「商品图 + 印刷实体 + bbox」视图
# ---------------------------------------------------------------------------
def _region_positions(boxes: List[List[int]]) -> Optional[List[str]]:
    """两处印刷图案时，返回各自相对另一处的方位词（偏上/偏下/偏左/偏右）。"""
    if len(boxes) != 2:
        return None
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    dy = centers[1][0] - centers[0][0]
    dx = centers[1][1] - centers[0][1]
    if abs(dy) >= abs(dx):
        positions = ["上", "下"] if dy > 0 else ["下", "上"]
    else:
        positions = ["左", "右"] if dx > 0 else ["右", "左"]
    return [f"偏{p}" for p in positions]


def build_views(gen: DatawheelV2Generator,
                train_records: List[Dict[str, Any]],
                name_maps: Dict[str, Dict[str, List[str]]],
                ref_map: Dict[Tuple[str, str], str],
                query_dir: Path,
                stats: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (每张商品的视图列表, stage_d 裁图信息)。视图按记录 sample_id 升序稳定排列。"""
    views: List[Dict[str, Any]] = []
    crop_records: List[Dict[str, Any]] = []

    def entity_info(domain: str, stem: str) -> Optional[Dict[str, Any]]:
        key = (domain, stem)
        if key not in ref_map:
            return None
        info = gen._entity(domain, stem)
        return {
            "domain": domain,
            "stem": stem,
            "full_title_zh": info.get("full_title_zh") or info.get("entity_name_zh") or stem,
            "entity_name_zh": info.get("entity_name_zh") or stem,
        }

    def make_view(kind: str, record: Dict[str, Any], ent: Dict[str, Any], box: List[int],
                  query_rel: str, image_key: str,
                  region: Optional[str] = None,
                  banned: Optional[List[Tuple[str, str]]] = None,
                  hint: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        return {
            "kind": kind,
            "source_stage": record["task_stage"],
            "source_sample_id": record["sample_id"],
            "query_image": query_rel,
            "image_key": image_key,
            "target_domain": ent["domain"],
            "target_stem": ent["stem"],
            "target_full_title_zh": ent["full_title_zh"],
            "target_name_zh": ent["entity_name_zh"],
            "bbox": list(box),
            "region": region,
            "banned": banned or [],
            "hint_negative": hint,
            "variant": False,
        }

    # ---------- stage_d：裁掉顶部参考带 ----------
    d_records = sorted((r for r in train_records if r["task_stage"] == "stage_d_reference_match"),
                       key=lambda r: r["sample_id"])
    for record in d_records:
        ents = _parse_entities(record)
        boxes = _parse_boxes(record)
        if len(ents) != 1 or len(boxes) != 1:
            stats["skipped_reasons"]["stage_d_bad_entity_or_bbox"] += 1
            continue
        ent = entity_info(ents[0]["domain"], ents[0]["stem"])
        if ent is None:
            stats["skipped_reasons"]["target_ref_render_unavailable"] += 1
            continue
        box = map_bbox_to_crop(boxes[0])
        if not bbox_valid(box):
            stats["skipped_reasons"]["stage_d_crop_bbox_invalid"] += 1
            continue
        src_path = PROJECT_ROOT / record["image_path"]
        if not src_path.is_file():
            stats["skipped_reasons"]["source_image_missing"] += 1
            continue
        src_image = Image.open(src_path).convert("RGB")
        if src_image.size != (CANVAS, D_HEIGHT):
            stats["skipped_reasons"]["stage_d_unexpected_image_size"] += 1
            continue
        product = src_image.crop((0, D_BAND, CANVAS, D_HEIGHT))
        q_filename = f"{record['sample_id']}_product.jpg"
        q_path = query_dir / q_filename
        product.save(q_path, format="JPEG", quality=92)
        query_rel = str(q_path.relative_to(PROJECT_ROOT))
        crop_records.append({"source_sample_id": record["sample_id"], "query_image": query_rel,
                             "bbox_full_1000": boxes[0], "bbox_crop_1000": box})
        views.append(make_view("d", record, ent, box, query_rel,
                               image_key=f"v2:{record['image_path']}"))
        stats["source_records_used"]["stage_d_reference_match"] += 1

    # ---------- stage_b：单图元 ----------
    b_records = sorted((r for r in train_records if r["task_stage"] == "stage_b_single_element"),
                       key=lambda r: r["sample_id"])
    for record in b_records:
        ents = _parse_entities(record)
        boxes = _parse_boxes(record)
        if len(ents) != 1 or len(boxes) != 1:
            stats["skipped_reasons"]["stage_b_bad_entity_or_bbox"] += 1
            continue
        ent = entity_info(ents[0]["domain"], ents[0]["stem"])
        if ent is None:
            stats["skipped_reasons"]["target_ref_render_unavailable"] += 1
            continue
        if not bbox_valid(boxes[0]):
            stats["skipped_reasons"]["stage_b_bbox_invalid"] += 1
            continue
        if not (PROJECT_ROOT / record["image_path"]).is_file():
            stats["skipped_reasons"]["source_image_missing"] += 1
            continue
        views.append(make_view("b", record, ent, boxes[0], record["image_path"],
                               image_key=f"v2:{record['image_path']}"))
        stats["source_records_used"]["stage_b_single_element"] += 1

    # ---------- stage_c：两图元，用相对方位消歧；同一商品图生成 2 个视图 ----------
    c_records = sorted((r for r in train_records if r["task_stage"] == "stage_c_multi_element"),
                       key=lambda r: r["sample_id"])
    for record in c_records:
        ents = _parse_entities(record)
        boxes = _parse_boxes(record)
        if len(ents) != len(boxes) or len(ents) < 1:
            stats["skipped_reasons"]["stage_c_bad_entity_or_bbox"] += 1
            continue
        if not (PROJECT_ROOT / record["image_path"]).is_file():
            stats["skipped_reasons"]["source_image_missing"] += 1
            continue
        entities: List[Dict[str, Any]] = []
        good_boxes: List[List[int]] = []
        good_indices: List[int] = []
        for idx, (ent_raw, box) in enumerate(zip(ents, boxes)):
            ent = entity_info(ent_raw["domain"], ent_raw["stem"])
            if ent is None or not bbox_valid(box):
                continue
            entities.append(ent)
            good_boxes.append(list(box))
            good_indices.append(idx)
        if not entities:
            stats["skipped_reasons"]["stage_c_no_valid_entity"] += 1
            continue
        banned_all = [(e["domain"], e["stem"]) for e in entities]
        if len(entities) == 1:
            # 极少见的单图元兜底记录（原 _gen_stage_b 兜底），按单图案视图使用
            views.append(make_view("c", record, entities[0], good_boxes[0], record["image_path"],
                                   image_key=f"v2:{record['image_path']}"))
        else:
            regions = _region_positions(good_boxes)
            if regions is None:
                stats["skipped_reasons"]["stage_c_region_ambiguity"] += 1
                continue
            for local_i, ent in enumerate(entities):
                others = [b for j, b in enumerate(banned_all) if j != local_i]
                view = make_view("c", record, ent, good_boxes[local_i], record["image_path"],
                                 image_key=f"v2:{record['image_path']}",
                                 region=regions[local_i], banned=others)
                views.append(view)
        stats["source_records_used"]["stage_c_multi_element"] += 1

    # ---------- stage_e：hard_distractor，商品上实际印刷的是干扰图元 ----------
    e_records = sorted((r for r in train_records if r["task_stage"] == "stage_e_negative_evidence"),
                       key=lambda r: r["sample_id"])
    for record in e_records:
        boxes = _parse_boxes(record)
        printed_ents = _parse_entities_json(record)
        queried = _parse_entities(record)
        if len(boxes) != 1 or not printed_ents or len(queried) != 1:
            # 该 E 记录没有印刷实体（absent_clean / occluded），不适合本任务
            stats["skipped_reasons"]["stage_e_no_printed_entity"] += 1
            continue
        domain = record["domain"].split("+")[0]
        printed = printed_ents[0]
        full_title = printed.get("full_title_zh") or ""
        matched = name_maps.get(domain, {}).get(full_title, []) if full_title else []
        if len(matched) != 1:
            stats["skipped_reasons"]["stage_e_distractor_unresolved"] += 1
            continue
        stem = matched[0]
        if _md5_bucket(f"{domain}/{stem}") >= 80:
            # 商品上印的实体属于 held-out 实体池 → 不作为 V6 训练图（split 卫生）
            stats["skipped_reasons"]["stage_e_distractor_held_out"] += 1
            continue
        ent = entity_info(domain, stem)
        if ent is None:
            stats["skipped_reasons"]["target_ref_render_unavailable"] += 1
            continue
        box = boxes[0]
        if not bbox_valid(box):
            stats["skipped_reasons"]["stage_e_bbox_invalid"] += 1
            continue
        if not (PROJECT_ROOT / record["image_path"]).is_file():
            stats["skipped_reasons"]["source_image_missing"] += 1
            continue
        hint = None
        hint_domain, hint_stem = queried[0]["domain"], queried[0]["stem"]
        if (hint_domain, hint_stem) in ref_map and (hint_domain, hint_stem) != (domain, stem):
            hint = (hint_domain, hint_stem)  # 与商品实际图案高度相似、但未印在图上 → 极难负例
        views.append(make_view("e", record, ent, box, record["image_path"],
                               image_key=f"v2:{record['image_path']}", hint=hint))
        stats["source_records_used"]["stage_e_negative_evidence"] += 1

    return views, crop_records


# ---------------------------------------------------------------------------
# 行生成
# ---------------------------------------------------------------------------
def _target_desc(view: Dict[str, Any]) -> str:
    if view.get("region"):
        return f"商品图上两处图案中{view['region']}的那处图案"
    return "商品上的图案"


def build_reference_row(view: Dict[str, Any],
                        label: str,
                        two_turn: bool,
                        row_index: int,
                        seed: int,
                        gen: DatawheelV2Generator,
                        ref_map: Dict[Tuple[str, str], str],
                        render_md5: Dict[Tuple[str, str], str],
                        stats: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(_seed_for(seed, "v6b_ref_row", row_index))
    domain, stem = view["target_domain"], view["target_stem"]
    banned = set(view["banned"])
    banned.add((domain, stem))  # 目标自身不能作为自己的负例

    # 与商品上已印图案渲染完全相同的参考图不能作候选（会导致答案歧义 / 标签噪声）
    forbidden_hashes = {render_md5[(domain, stem)]} if (domain, stem) in render_md5 else set()
    for banned_entity in view["banned"]:
        hash_value = render_md5.get(banned_entity)
        if hash_value is not None:
            forbidden_hashes.add(hash_value)

    n_refs = 3 if rng.random() < 0.55 else 2
    n_neg = n_refs - 1 if label == "match" else n_refs

    candidates: List[Dict[str, Any]] = []
    if label == "match":
        candidates.append({"domain": domain, "stem": stem, "positive": True, "distance": 0})
    if view.get("hint_negative") and n_neg > 0:
        hint_domain, hint_stem = view["hint_negative"]
        hint_hash = render_md5.get((hint_domain, hint_stem))
        if hint_hash is not None and hint_hash not in forbidden_hashes:
            candidates.append({
                "domain": hint_domain, "stem": hint_stem, "positive": False,
                "distance": dhash_distance(dhash_map_global, (domain, stem), (hint_domain, hint_stem)),
            })
            banned.add((hint_domain, hint_stem))
            forbidden_hashes.add(hint_hash)
            n_neg -= 1
    negatives = pick_hard_negatives(gen, domain, stem, n_neg, rng, banned, ref_map,
                                    render_md5, forbidden_hashes)
    candidates.extend(dict(item, positive=False) for item in negatives)

    if len(candidates) != n_refs:
        raise RuntimeError(f"candidate count {len(candidates)} != {n_refs} for {view['source_sample_id']}")
    if label == "match" and sum(1 for c in candidates if c["positive"]) != 1:
        raise RuntimeError("match row must contain exactly one positive candidate")

    rng.shuffle(candidates)
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    match_idx = next((i for i, c in enumerate(candidates) if c["positive"]), None)
    ref_rels = [ref_map[(c["domain"], c["stem"])] for c in candidates]
    images = [view["query_image"]] + ref_rels
    n_images = len(images)

    box_str = _json_bbox(view["bbox"])
    target_desc = _target_desc(view)
    tmpl = INSTRUCTION_TEMPLATES[rng.randrange(len(INSTRUCTION_TEMPLATES))]
    instruction = tmpl.format(k=n_refs, last=n_refs + 1, letters="/".join(letters), td=target_desc,
                              bbox_clause="" if two_turn else ONE_TURN_BBOX_CLAUSE)
    first_content = "<image>" * n_images + "\n" + instruction

    if label == "match":
        if match_idx is None:
            raise RuntimeError("match row without positive index")
        short_answer = f"与候选{letters[match_idx]}一致。"
    else:
        short_answer = f"没有候选参考图与商品图案一致；商品上的图案是【{view['target_full_title_zh']}】。"

    if two_turn:
        bbox_question = BBOX_QUESTION_TEMPLATES[rng.randrange(len(BBOX_QUESTION_TEMPLATES))]
        messages = [
            {"role": "user", "content": first_content},
            {"role": "assistant", "content": short_answer},
            {"role": "user", "content": bbox_question},
            {"role": "assistant", "content": box_str},
        ]
    else:
        final_answer = short_answer + f"商品上该图案的归一化边界框为 {box_str}。"
        messages = [
            {"role": "user", "content": first_content},
            {"role": "assistant", "content": final_answer},
        ]

    # ---- 契约自检 ----
    placeholder_total = sum(m["content"].count("<image>") for m in messages)
    if placeholder_total != n_images or messages[0]["content"].count("<image>") != n_images:
        raise RuntimeError(f"<image> placeholder mismatch: {placeholder_total} vs {n_images}")
    text = "\n".join(m["content"] for m in messages)
    length_est = estimate_text_tokens(text) + n_images * IMG_EST_TOKENS
    if length_est > MAX_LEN:
        raise RuntimeError(f"estimated length {length_est:.1f} > {MAX_LEN}")

    # ---- 统计 ----
    stats["n_images"].append(n_images)
    stats["n_refs"].append(n_refs)
    stats["labels"][label] += 1
    stats["domains"][view["target_domain"]] += 1
    stats["source_stages"][view["source_stage"]] += 1
    stats["query_modes"]["stage_d_crop" if view["kind"] == "d" else "v2_direct"] += 1
    stats["regions"]["c_region" if view.get("region") else "single_pattern"] += 1
    stats["two_turn_rows"] += int(two_turn)
    stats["lengths"].append(length_est)
    stats["text_tokens"].append(estimate_text_tokens(text))
    stats["bbox_areas"].append(bbox_area_ratio(view["bbox"]))
    stats["negative_distances"].extend(
        c["distance"] for c in candidates if not c["positive"] and c.get("distance") is not None)
    for c in candidates:
        stats["candidate_domains"][c["domain"]] += 1
        stats["unique_refs"].add(ref_map[(c["domain"], c["stem"])])
    stats["query_images"].add(view["query_image"])
    stats["views_per_image"][view["image_key"]] = stats["views_per_image"].get(view["image_key"], 0) + 1

    return {
        "messages": messages,
        "images": images,
        "enable_thinking": False,
        "sample_id": "",  # 由 main 统一编号
        "cate_id": "reference_match",
        "variant": VARIANT,
        "_label": label,
        "_source_sample_id": view["source_sample_id"],
        "_source_stage": view["source_stage"],
        "_query_image": view["query_image"],
        "_target_domain": domain,
        "_target_stem": stem,
        "_target_full_title_zh": view["target_full_title_zh"],
        "_target_bbox": list(view["bbox"]),
        "_letters": letters,
        "_match_idx": match_idx,
        "_n_refs": n_refs,
        "_two_turn": two_turn,
        "_length_est": length_est,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_dataset(rows_target: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    global dhash_map_global
    started = time.time()
    out_dir = out_dir if out_dir.is_absolute() else (PROJECT_ROOT / out_dir)
    parts_dir = out_dir / "parts"
    query_dir = out_dir / "images" / "query"
    ref_dir = out_dir / "images" / "refs"
    for path in (parts_dir, query_dir, ref_dir):
        path.mkdir(parents=True, exist_ok=True)
    global REF_DIR
    REF_DIR = ref_dir

    if rows_target < 1:
        raise ValueError("--rows must be positive")
    if not V2_DATASET.is_file():
        raise FileNotFoundError(f"missing v2 dataset: {V2_DATASET}")
    v2_sha256 = _sha256_file(V2_DATASET)

    print(f"[reference] loading {V2_DATASET.relative_to(PROJECT_ROOT)} ...")
    all_records = pq.read_table(V2_DATASET).to_pylist()
    train_records = [r for r in all_records if r["split"] == "train"]
    print(f"[reference] v2 rows={len(all_records)} train={len(train_records)} (split=train only)")

    gen = DatawheelV2Generator(seed)
    ref_map, dhash_map, render_md5, ref_failures = build_asset_tables(gen)
    dhash_map_global = dhash_map
    duplicate_groups: Dict[str, List[str]] = defaultdict(list)
    for (dup_domain, dup_stem), hash_value in render_md5.items():
        duplicate_groups[hash_value].append(f"{dup_domain}__{dup_stem}")
    render_duplicate_groups = sorted(sorted(group) for group in duplicate_groups.values() if len(group) > 1)
    if render_duplicate_groups:
        print(f"[reference] texture-identical reference groups excluded from cross-candidates: "
              f"{render_duplicate_groups}")

    # 名称表：仅供 stage_e 从 entities_json 反查印刷干扰图元的 stem
    name_maps: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for domain, stems in gen.asset_domains.items():
        for stem in stems:
            info = gen._entity(domain, stem)
            title = info.get("full_title_zh")
            if title:
                name_maps[domain][title].append(stem)

    stats: Dict[str, Any] = {
        "skipped_reasons": Counter(),
        "source_records_used": Counter(),
    }
    print("[reference] building query views from v2 train records ...")
    views, crop_records = build_views(gen, train_records, name_maps, ref_map, query_dir, stats)

    # 按图像分组，保持稳定顺序（d → b → c → e，组内按 sample_id）
    images_in_order: List[str] = []
    views_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for view in views:
        if view["image_key"] not in views_by_image:
            views_by_image[view["image_key"]] = []
            images_in_order.append(view["image_key"])
        views_by_image[view["image_key"]].append(view)

    base_views: List[Dict[str, Any]] = []
    secondary_pool: List[Dict[str, Any]] = []
    for image_key in images_in_order:
        candidates = views_by_image[image_key]
        if len(candidates) >= 2:
            candidates = sorted(candidates, key=lambda v: (-bbox_area_ratio(v["bbox"]), v["target_stem"]))
            base_views.append(candidates[0])
            secondary_pool.extend(candidates[1:])
        else:
            base_views.append(candidates[0])

    rng = random.Random(_seed_for(seed, "v6b_reference_selection"))
    if rows_target <= len(base_views):
        rng.shuffle(base_views)
        selected = base_views[:rows_target]
        extras: List[Dict[str, Any]] = []
        multi_view_rows = 0
    else:
        selected = list(base_views)
        extras_needed = rows_target - len(base_views)
        rng.shuffle(secondary_pool)
        extras = secondary_pool[:extras_needed]
        if len(extras) < extras_needed:
            # 兜底：用 D 阶段同一商品图的不同候选集/问法视图补足（每图仍 ≤ 3 行）
            variant_pool = [dict(v, variant=True) for v in base_views if v["kind"] == "d"]
            variant_pool += [dict(v, variant=True) for v in base_views if v["kind"] != "d"]
            rng.shuffle(variant_pool)
            extras.extend(variant_pool[: extras_needed - len(extras)])
        if len(extras) < extras_needed:
            raise RuntimeError(f"not enough query views: base={len(base_views)} extra={len(extras)} "
                               f"need={rows_target}")
        multi_view_rows = len(extras)

    rows_spec = selected + extras
    if len(rows_spec) != rows_target:
        raise RuntimeError(f"row spec count {len(rows_spec)} != {rows_target}")

    # ---- 70/30 标签分配：多视图成对做 match/no_match 对照，其余随机补齐 ----
    by_image: Dict[str, List[int]] = defaultdict(list)
    for idx, view in enumerate(rows_spec):
        by_image[view["image_key"]].append(idx)
    labels: List[Optional[str]] = [None] * rows_target
    paired = 0
    for idxs in by_image.values():
        if len(idxs) == 2:
            paired += 1
            if rng.random() < 0.5:
                labels[idxs[0]], labels[idxs[1]] = "match", "no_match"
            else:
                labels[idxs[0]], labels[idxs[1]] = "no_match", "match"
        elif len(idxs) > 2:
            raise RuntimeError("multi-view rows per image must be <= 2 in current allocation")
    singles = [i for i, label in enumerate(labels) if label is None]
    match_total = int(round(rows_target * MATCH_RATIO))
    no_match_total = rows_target - match_total
    match_single = match_total - paired
    no_match_single = no_match_total - paired
    if match_single < 0 or no_match_single < 0 or match_single + no_match_single != len(singles):
        raise RuntimeError(f"label quota infeasible: paired={paired} singles={len(singles)} "
                           f"match={match_total} no_match={no_match_total}")
    rng.shuffle(singles)
    for i in singles[:match_single]:
        labels[i] = "match"
    for i in singles[match_single:]:
        labels[i] = "no_match"
    if labels.count("match") != match_total or labels.count("no_match") != no_match_total:
        raise RuntimeError("label quota mismatch")
    if any(label is None for label in labels):
        raise RuntimeError("unlabeled rows remain")

    # ---- 50% 两轮 ----
    two_turn_true = (rows_target + 1) // 2
    two_turn_flags = [True] * two_turn_true + [False] * (rows_target - two_turn_true)
    rng.shuffle(two_turn_flags)
    if sum(two_turn_flags) != two_turn_true:
        raise RuntimeError("two-turn quota mismatch")

    # ---- 打乱行序后生成，sample_id 按输出顺序编号 ----
    order = list(range(rows_target))
    rng.shuffle(order)
    long_stats: Dict[str, Any] = {
        "n_images": [], "n_refs": [], "labels": Counter(), "domains": Counter(),
        "source_stages": Counter(), "query_modes": Counter(), "regions": Counter(),
        "two_turn_rows": 0, "lengths": [], "text_tokens": [], "bbox_areas": [],
        "negative_distances": [], "candidate_domains": Counter(),
        "unique_refs": set(), "query_images": set(), "views_per_image": Counter(),
    }
    rows: List[Dict[str, Any]] = []
    for out_idx, spec_idx in enumerate(order, start=1):
        row = build_reference_row(rows_spec[spec_idx], labels[spec_idx], two_turn_flags[spec_idx],
                                  out_idx, seed, gen, ref_map, render_md5, long_stats)
        row["sample_id"] = f"V6B_R_{out_idx:07d}"
        rows.append(row)

    # ---- 契约复检（与其他 V6 part 一致）----
    sample_ids = [r["sample_id"] for r in rows]
    if len(set(sample_ids)) != rows_target or len({r["sample_id"] for r in rows}) != rows_target:
        raise RuntimeError("sample_id not unique")
    row_keys = {(r["sample_id"], r["cate_id"], tuple(r["images"])) for r in rows}
    if len(row_keys) != rows_target:
        raise RuntimeError("row-level key (sample_id, cate_id, images) not unique")
    if any(r["_label"] not in ("match", "no_match") for r in rows):
        raise RuntimeError("unexpected label")
    if sum(r["_length_est"] for r in rows) <= 0 or max(r["_length_est"] for r in rows) > MAX_LEN:
        raise RuntimeError("length check failed")

    clean_rows = [{k: r[k] for k in ("messages", "images", "enable_thinking", "sample_id", "cate_id", "variant")}
                  for r in rows]
    table = pa.Table.from_pylist(clean_rows, schema=SCHEMA)
    parquet_path = parts_dir / "reference_rows.parquet"
    pq.write_table(table, parquet_path, compression="zstd")
    parquet_sha256 = _sha256_file(parquet_path)
    print(f"[reference] wrote {parquet_path.relative_to(PROJECT_ROOT)} "
          f"({parquet_path.stat().st_size / 1024:.0f} KB, {table.num_rows} rows)")

    match_rows = long_stats["labels"]["match"]
    no_match_rows = long_stats["labels"]["no_match"]
    image_count_dist = {str(n): long_stats["n_images"].count(n) for n in sorted(set(long_stats["n_images"]))}
    ref_count_dist = {str(n): long_stats["n_refs"].count(n) for n in sorted(set(long_stats["n_refs"]))}
    views_per_image_dist = {str(n): sum(1 for c in long_stats["views_per_image"].values() if c == n)
                            for n in sorted(set(long_stats["views_per_image"].values()))}
    source_stage_dist = {stage: long_stats["source_stages"][stage]
                         for stage in SOURCE_STAGES if long_stats["source_stages"][stage]}
    domain_dist = {domain: long_stats["domains"][domain] for domain in sorted(long_stats["domains"])}
    candidate_domain_dist = {domain: long_stats["candidate_domains"][domain]
                             for domain in sorted(long_stats["candidate_domains"])}
    skipped = {reason: int(count) for reason, count in sorted(stats["skipped_reasons"].items()) if count}
    source_used = {stage: int(stats["source_records_used"][stage])
                   for stage in SOURCE_STAGES if stats["source_records_used"][stage]}

    # 示例行：match / no_match 各取 1 行（利于快速目检两条分支）
    examples: List[Dict[str, Any]] = []
    for want_label in ("match", "no_match"):
        for row in rows:
            if row["_label"] != want_label:
                continue
            examples.append({
                "sample_id": row["sample_id"], "images": row["images"], "messages": row["messages"],
                "meta": {"label": row["_label"], "source_sample_id": row["_source_sample_id"],
                         "source_stage": row["_source_stage"], "target_bbox_1000": row["_target_bbox"],
                         "target_full_title_zh": row["_target_full_title_zh"],
                         "n_refs": row["_n_refs"], "two_turn": row["_two_turn"],
                         "length_est": round(row["_length_est"], 1)}})
            break

    text_tokens = long_stats["text_tokens"]
    total_tokens = long_stats["lengths"]
    manifest = {
        "part": "reference_rows",
        "tool": "tools/build_v6_balanced_reference.py",
        "format": "ms-swift (messages/images/enable_thinking/sample_id/cate_id/variant)",
        "cate_id": "reference_match",
        "variant": VARIANT,
        "enable_thinking": False,
        "seed": seed,
        "deterministic": True,
        "rows": rows_target,
        "split_by_match": {
            "match": match_rows,
            "no_match": no_match_rows,
            "match_ratio": round(match_rows / rows_target, 4),
            "target_ratio": MATCH_RATIO,
        },
        "image_count_dist": image_count_dist,
        "ref_count_dist": ref_count_dist,
        "views_per_query_image_dist": views_per_image_dist,
        "multi_view_rows": multi_view_rows,
        "multi_view_images": sum(1 for c in long_stats["views_per_image"].values() if c >= 2),
        "source_stage_dist": source_stage_dist,
        "source_records_used": source_used,
        "domain_dist": domain_dist,
        "candidate_domain_dist": candidate_domain_dist,
        "query_mode_dist": dict(sorted(long_stats["query_modes"].items())),
        "target_region_mode_dist": dict(sorted(long_stats["regions"].items())),
        "two_turn_rows": long_stats["two_turn_rows"],
        "two_turn_ratio": round(long_stats["two_turn_rows"] / rows_target, 4),
        "unique_query_images": len(long_stats["query_images"]),
        "unique_ref_images_used": len(long_stats["unique_refs"]),
        "refs_rendered_total": len(ref_map),
        "render_failures": ref_failures,
        "total_image_refs": sum(long_stats["n_images"]),
        "image_reuse_rows": rows_target - len(long_stats["query_images"]),
        "query_image_dir": "images/query",
        "ref_image_dir": "images/refs",
        "stage_d_crops": len(crop_records),
        "bbox_area_ratio_stats": _stats(long_stats["bbox_areas"], 6),
        "negative_dhash_distance_stats": _stats(long_stats["negative_distances"], 2),
        "length_est": {
            "formula": "CJK chars/1.6 + non-CJK chars/4 + images*512",
            "text_tokens": _stats(text_tokens, 2),
            "total_tokens_est": _stats(total_tokens, 2),
            "total_tokens_est_sum": round(sum(total_tokens), 1),
            "max_len": MAX_LEN,
            "over_max_len": sum(1 for v in total_tokens if v > MAX_LEN),
        },
        "source": {
            "v2_dataset": "output/datawheel_v2/dataset.parquet",
            "v2_dataset_sha256": v2_sha256,
            "v2_split": "train",
            "v2_train_rows": len(train_records),
            "candidate_pool_rule": "train-visible entities only (_md5_bucket(domain/stem) < 80)",
            "render_identical_reference_groups": render_duplicate_groups,
            "skipped": skipped,
            "base_unique_query_images": len(base_views),
            "extra_views_needed": max(0, rows_target - len(base_views)),
        },
        "checks": {
            "row_count_exact": len(rows) == rows_target,
            "sample_id_unique": len({r["sample_id"] for r in rows}) == rows_target,
            "row_key_unique": len(row_keys) == rows_target,
            "placeholder_matches_images": all(
                sum(m["content"].count("<image>") for m in r["messages"]) == len(r["images"])
                and r["messages"][0]["content"].count("<image>") == len(r["images"])
                for r in rows),
            "first_user_holds_all_placeholders": True,
            "role_alternation": all(
                all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(r["messages"]))
                for r in rows),
            "variant_all_6": all(r["variant"] == VARIANT for r in rows),
            "enable_thinking_all_false": all(r["enable_thinking"] is False for r in rows),
            "est_length_max_under_8192": max(total_tokens) <= MAX_LEN,
            "match_rows_have_positive_candidate": all(
                r["_label"] != "match" or r["_match_idx"] is not None for r in rows),
            "no_match_rows_have_no_positive_candidate": all(
                r["_label"] != "no_match" or r["_match_idx"] is None for r in rows),
        },
        "parquet": {
            "path": "output/datawheel_v6_balanced/parts/reference_rows.parquet",
            "sha256": parquet_sha256,
            "compression": "zstd",
            "bytes": parquet_path.stat().st_size,
        },
        "examples": examples,
    }
    if not all(manifest["checks"].values()):
        failed = [name for name, ok in manifest["checks"].items() if not ok]
        raise RuntimeError(f"contract checks failed: {failed}")

    manifest_path = parts_dir / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
                             encoding="utf-8")
    elapsed = round(time.time() - started, 1)
    print(f"[reference] manifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"[reference] rows={rows_target} match={match_rows} ({match_rows / rows_target:.3f}) "
          f"no_match={no_match_rows} images/row={image_count_dist} "
          f"two_turn={long_stats['two_turn_rows']} unique_query={len(long_stats['query_images'])} "
          f"unique_refs_used={len(long_stats['unique_refs'])} refs_rendered={len(ref_map)}")
    print(f"[reference] max est tokens={max(total_tokens):.1f} (512/img) "
          f"mean={sum(total_tokens) / len(total_tokens):.1f}; elapsed={elapsed}s")
    print("[reference] done.")
    return manifest


REF_DIR = DEFAULT_OUT_DIR / "images" / "refs"  # 由 build_dataset 覆盖


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build V6 balanced multi-image reference matching rows")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help=f"目标行数（默认 {DEFAULT_ROWS}，cate_id=reference_match）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（确定性）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="输出根目录（默认 output/datawheel_v6_balanced）")
    args = parser.parse_args(argv)

    manifest = build_dataset(args.rows, args.seed, args.out_dir)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in ("examples",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
