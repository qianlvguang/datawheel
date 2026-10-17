#!/usr/bin/env python3
"""Datawheel v2 → V6 中训练参考图匹配行 (multi-image reference matching rows)。

把 v2 的 stage_d 复合图拆成「多图 in-context 样本」，完全对齐 Qwen3.5-27B 中训练
V1–V5 的 ms-swift 数据格式：
  columns = messages / images / enable_thinking / sample_id / cate_id / variant

每条 row：
- 第 1 张图：从 v2 stage_d 复合图中裁出的 640x640 商品实拍图（去掉原参考图带）；
- 后续 2~3 张图：印刷图元的标准参考渲染图（白底 JPEG），其中匹配样本包含正例，
  不匹配样本全部为 dHash 视觉近邻困难负例；
- messages：首轮 user 含 N 个 <image> + 判断指令；assistant 给出候选字母；
  50% 样本再追加一轮 bbox grounding 问答。
- variant=6（训练插件自动映射为 loss_V6），enable_thinking=False。

产物：
  output/datawheel_v6_pilot/parts/reference_multi.parquet
  output/datawheel_v6_pilot/parts/reference_manifest.json
  output/datawheel_v6_pilot/images/query/*.jpg
  output/datawheel_v6_pilot/images/refs/*.jpg

用法：
  .venv/bin/python tools/build_v6_reference_rows.py --limit 650
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_datawheel_v2 import DatawheelV2Generator  # noqa: E402

V2_DATASET = PROJECT_ROOT / "output" / "datawheel_v2" / "dataset.parquet"
OUT_DIR = PROJECT_ROOT / "output" / "datawheel_v6_pilot"
PARTS_DIR = OUT_DIR / "parts"
QUERY_DIR = OUT_DIR / "images" / "query"
REF_DIR = OUT_DIR / "images" / "refs"

CANVAS = 640
D_BAND = 200
D_HEIGHT = 840
REF_SIZE = 384
VARIANT = 6

MESSAGES_SCHEMA = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("messages", MESSAGES_SCHEMA),
    ("images", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
    ("sample_id", pa.string()),
    ("cate_id", pa.string()),
    ("variant", pa.int32()),
])


def map_bbox_to_crop(box: List[int]) -> List[int]:
    """v2 640x840 全图 [ymin,xmin,ymax,xmax] → 裁掉顶部 200px 后 640x640 商品图坐标。"""
    ymin, xmin, ymax, xmax = box
    ny_min = int(round(((ymin * D_HEIGHT / 1000.0) - D_BAND) / CANVAS * 1000))
    ny_max = int(round(((ymax * D_HEIGHT / 1000.0) - D_BAND) / CANVAS * 1000))
    return [max(0, min(1000, ny_min)), max(0, min(1000, xmin)),
            max(0, min(1000, ny_max)), max(0, min(1000, xmax))]


def render_reference(gen: DatawheelV2Generator, domain: str, stem: str,
                     cache: Dict[str, str]) -> Optional[str]:
    """把图元渲染成白底 384x384 JPEG 参考图，返回项目相对路径。"""
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
    rel = str(out_path.relative_to(PROJECT_ROOT))
    cache[key] = rel
    return rel


def build_rows(limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    rows = pq.read_table(V2_DATASET).to_pylist()
    d_rows = [r for r in rows if r["task_stage"] == "stage_d_reference_match"
              and r["split"] == "train"  # V6 仅用训练 split，避免把 v2 泛化评测集泄入中训练
              and r["image_path"] and (PROJECT_ROOT / r["image_path"]).exists()]
    rng = random.Random(seed)
    d_rows = d_rows[:]
    rng.shuffle(d_rows)
    d_rows = d_rows[:limit]

    gen = DatawheelV2Generator(seed)
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_cache: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []
    skipped = 0

    for i, rec in enumerate(d_rows):
        try:
            ent = json.loads(rec["entity_ids_json"])[0]
            domain, stem = ent["domain"], ent["stem"]
            full_box = json.loads(rec["bboxes_norm_1000"])[0]
            box_crop = map_bbox_to_crop(full_box)
            if box_crop[0] >= box_crop[2] or box_crop[1] >= box_crop[3]:
                skipped += 1
                continue
            printed = gen._entity(domain, stem)

            # 1) 裁出商品图
            src = Image.open(PROJECT_ROOT / rec["image_path"]).convert("RGB")
            product = src.crop((0, D_BAND, CANVAS, D_HEIGHT))
            qid = f"V6R_{i + 1:06d}"
            q_path = QUERY_DIR / f"{qid}_product.jpg"
            product.save(q_path, format="JPEG", quality=92)
            query_rel = str(q_path.relative_to(PROJECT_ROOT))

            # 2) 候选参考图：匹配样本 = 正例 + 困难负例；不匹配样本 = 全负例
            case_match = rec["match_label"] == "match"
            n_refs = 3 if rng.random() < 0.6 else 2
            neg_count = n_refs - 1 if case_match else n_refs
            negatives = gen._hard_negatives(domain, stem, neg_count, rng)
            cand: List[Dict[str, Any]] = []
            if case_match:
                cand.append({"domain": domain, "stem": stem, "positive": True})
            for s in negatives:
                cand.append({"domain": domain, "stem": s, "positive": False})
            rng.shuffle(cand)
            letters = [chr(ord("A") + k) for k in range(len(cand))]
            match_idx = next((k for k, c in enumerate(cand) if c["positive"]), None)
            if case_match and match_idx is None:
                skipped += 1
                continue

            ref_rels: List[str] = []
            for c in cand:
                rel = render_reference(gen, c["domain"], c["stem"], ref_cache)
                if rel is None:
                    break
                ref_rels.append(rel)
            if len(ref_rels) != len(cand):
                skipped += 1
                continue

            images = [query_rel] + ref_rels
            n = len(images)
            placeholders = "<image>" * n
            if case_match:
                answer1 = f"与候选{letters[match_idx]}一致。"
            else:
                answer1 = f"没有候选参考图与商品图案一致；商品上的图案是【{printed['full_title_zh']}】。"
            prompt1 = (f"{placeholders}\n第一张为商品实拍图，后续 {len(cand)} 张为候选参考图"
                       f"（{'/'.join(letters)}）。请判断商品上的图案与哪一张候选参考图一致；"
                       f"若均不一致，请明确说明并指出商品上的图案。")

            two_turn = rng.random() < 0.5
            if two_turn:
                messages = [
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": answer1},
                    {"role": "user", "content": "请给出商品上该图案的归一化边界框 [ymin,xmin,ymax,xmax]。"},
                    {"role": "assistant", "content": str(box_crop)},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": f"{answer1}商品上该图案的归一化边界框为 {box_crop}。"},
                ]
            records.append({
                "messages": messages,
                "images": images,
                "enable_thinking": False,
                "sample_id": qid,
                "cate_id": "reference_match",
                "variant": VARIANT,
                "_case_match": case_match,
                "_match_idx": match_idx,
                "_letters": letters,
                "_n_images": n,
                "_domain": domain,
                "_bbox_crop": box_crop,
            })
        except Exception as exc:  # 单条失败不阻塞
            skipped += 1
            print(f"[skip] {rec['sample_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"[reference] built {len(records)} rows, skipped {skipped}, refs cache {len(ref_cache)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V6 multi-image reference matching rows")
    parser.add_argument("--limit", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_rows(args.limit, args.seed)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    clean = [{k: r[k] for k in ("messages", "images", "enable_thinking", "sample_id", "cate_id", "variant")}
             for r in records]
    table = pa.Table.from_pylist(clean, schema=SCHEMA)
    out_parquet = PARTS_DIR / "reference_multi.parquet"
    pq.write_table(table, out_parquet, compression="zstd")
    print(f"[reference] wrote {out_parquet} ({out_parquet.stat().st_size / 1024:.0f} KB, {table.num_rows} rows)")

    manifest = {
        "part": "reference_multi",
        "rows": len(records),
        "split_by_match": {
            "match": sum(1 for r in records if r["_case_match"]),
            "no_match": sum(1 for r in records if not r["_case_match"]),
        },
        "image_count_dist": {str(n): sum(1 for r in records if r["_n_images"] == n)
                             for n in sorted({r["_n_images"] for r in records})},
        "ref_count_dist": {str(n - 1): sum(1 for r in records if r["_n_images"] == n)
                            for n in sorted({r["_n_images"] for r in records})},
        "domain_dist": {d: sum(1 for r in records if r["_domain"] == d)
                        for d in sorted({r["_domain"] for r in records})},
        "two_turn_ratio": round(sum(1 for r in records if len(r["messages"]) == 4) / max(len(records), 1), 3),
        "bbox_area_ratio_stats": {
            "min": round(min((b[2] - b[0]) * (b[3] - b[1]) / 1_000_000 for b in (r["_bbox_crop"] for r in records)), 4),
            "median": round(sorted((b[2] - b[0]) * (b[3] - b[1]) / 1_000_000
                                   for b in (r["_bbox_crop"] for r in records))[len(records) // 2], 4),
        },
        "examples": [{"sample_id": r["sample_id"], "images": r["images"],
                      "messages": r["messages"][:2]} for r in records[:2]],
    }
    (PARTS_DIR / "reference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "examples"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
