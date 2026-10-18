#!/usr/bin/env python3
"""M1：Datawheel 现有数据摆放审计。

检查：
- 图案 bbox 是否落在该商品的可印区（substrate zone）内；
- bbox 中心是否"塌缩"到图像中心，而偏离该商品的可印区中心；
- 按阶段/品类输出指标，并生成少量叠加图供人工核验。

不修改数据，只产出审计报告。Stage A（合成色块）与 Stage D（参考图版式）默认排除。

用法：
  .venv/bin/python tools/placement_audit.py \
    --dataset output/datawheel_v2/dataset.parquet \
    --out output/audit/placement
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from placement_common import (
    PROJECT_ROOT, bbox_to_pixels, canvas_zone_pixel_rect, find_zone_spec,
    load_substrates, zone_pixel_rect,
)

DEFAULT_DATASET = PROJECT_ROOT / "output" / "datawheel_v2" / "dataset.parquet"
DEFAULT_OUT = PROJECT_ROOT / "output" / "audit" / "placement"


def normalized_center(bbox: List[int]) -> Tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return ((x0 + x1) / 2000.0, (y0 + y1) / 2000.0)


def norm_rect(rect_px: Tuple[int, int, int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect_px
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def intersection_area_ratio(bbox: List[int], zone_rect: Tuple[int, int, int, int],
                            width: int, height: int) -> float:
    x0, y0, x1, y1 = bbox_to_pixels(bbox, width, height)
    zx0, zy0, zx1, zy1 = zone_rect
    ix0, iy0 = max(x0, zx0), max(y0, zy0)
    ix1, iy1 = min(x1, zx1), min(y1, zy1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (x1 - x0) * (y1 - y0))
    return inter / area


def median(values: List[float]) -> Optional[float]:
    return round(statistics.median(values), 4) if values else None


def mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def draw_debug(rows: List[Dict[str, Any]], out_path: Path, max_tiles: int = 6) -> None:
    tiles: List[Image.Image] = []
    for item in rows[:max_tiles]:
        try:
            img = Image.open(PROJECT_ROOT / item["image_path"]).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        d = ImageDraw.Draw(img)
        # 图像中心
        cx, cy = w / 2, h / 2
        d.line([cx - 12, cy, cx + 12, cy], fill=(255, 0, 0), width=2)
        d.line([cx, cy - 12, cx, cy + 12], fill=(255, 0, 0), width=2)
        # 可印区
        zx0, zy0, zx1, zy1 = item["zone_rect_px"]
        d.rectangle([zx0, zy0, zx1, zy1], outline=(0, 180, 255), width=3)
        # 图案 bbox
        bx0, by0, bx1, by1 = bbox_to_pixels(item["bbox"], w, h)
        d.rectangle([bx0, by0, bx1, by1], outline=(255, 40, 40), width=3)
        d.text((4, 4), f'{item["sample_id"]} {item["stage"]}', fill=(0, 0, 0))
        tiles.append(img)
    if not tiles:
        return
    tw = min(640, tiles[0].width)
    th = int(tiles[0].height * tw / tiles[0].width)
    sheet = Image.new("RGB", (tw * 2 + 8, th * ((len(tiles) + 1) // 2) + 8 * ((len(tiles) + 1) // 2)),
                      (245, 245, 245))
    for i, tile in enumerate(tiles):
        tile = tile.resize((tw, th), Image.Resampling.LANCZOS)
        col, row = i % 2, i // 2
        sheet.paste(tile, (col * (tw + 8), row * (th + 8)))
    sheet.save(out_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Datawheel pattern placement vs product zones")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stages", nargs="*", default=["stage_b_single_element",
                                                          "stage_c_multi_element",
                                                          "stage_e_negative_evidence"])
    parser.add_argument("--center-threshold", type=float, default=0.06,
                        help="bbox 中心距图像中心小于该归一化距离视为 center collapse 候选")
    parser.add_argument("--expected-threshold", type=float, default=0.10,
                        help="当可印区中心距图像中心大于该距离，才检测 center collapse")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()

    rows = pq.read_table(args.dataset).to_pylist()
    substrates = load_substrates()
    stages = set(args.stages)
    issues: List[Dict[str, Any]] = []
    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_missing_substrate = 0
    skipped_no_bbox = 0

    for row in rows:
        stage = row.get("task_stage")
        if stage not in stages:
            continue
        sub = substrates.get(row.get("substrate_id", ""))
        if not sub:
            skipped_missing_substrate += 1
            continue
        try:
            bboxes = json.loads(row.get("bboxes_norm_1000") or "[]")
        except Exception:
            bboxes = []
        if not bboxes:
            skipped_no_bbox += 1
            continue
        width = int(row.get("image_width") or 640)
        height = int(row.get("image_height") or 640)
        category = sub.get("sub_category", "")
        # 可印区：优先使用具体 zone spec，C 布局则用整体 zone
        placement_zone = row.get("placement_zone") or ""
        try:
            spec = find_zone_spec(sub, placement_zone)
            zone_rect = zone_pixel_rect(sub, spec, width, height)
            expected_source = f"spec:{spec['id']}"
        except Exception:
            zone_rect = canvas_zone_pixel_rect(sub, width, height)
            expected_source = "canvas_zone"
        zone_center = norm_rect(zone_rect, width, height)
        zone_cx = (zone_center[0] + zone_center[2]) / 2
        zone_cy = (zone_center[1] + zone_center[3]) / 2
        zone_center_norm = (zone_cx, zone_cy)
        expected_center_dist = distance(zone_center_norm, (0.5, 0.5))

        for bbox in bboxes:
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            center = normalized_center(bbox)
            image_center_dist = distance(center, (0.5, 0.5))
            containment = intersection_area_ratio(bbox, zone_rect, width, height)
            center_collapse = bool(image_center_dist < args.center_threshold
                                   and expected_center_dist > args.expected_threshold)
            item = {
                "sample_id": row.get("sample_id"),
                "stage": stage,
                "category": category,
                "substrate_id": row.get("substrate_id"),
                "placement_zone": placement_zone,
                "expected_source": expected_source,
                "bbox": bbox,
                "bbox_center": [round(center[0], 4), round(center[1], 4)],
                "zone_center": [round(zone_center_norm[0], 4), round(zone_center_norm[1], 4)],
                "image_center_dist": round(image_center_dist, 4),
                "expected_center_dist": round(expected_center_dist, 4),
                "zone_containment": round(containment, 4),
                "center_collapse": center_collapse,
                "image_path": row.get("image_path"),
                "zone_rect_px": list(zone_rect),
            }
            by_stage[stage].append(item)
            by_category[category].append(item)
            if center_collapse or containment < 0.9:
                issues.append(item)

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        collapses = sum(1 for x in items if x["center_collapse"])
        return {
            "bboxes": total,
            "center_collapse": collapses,
            "center_collapse_rate": round(collapses / total, 4) if total else 0,
            "zone_containment_median": median([x["zone_containment"] for x in items]),
            "image_center_dist_median": median([x["image_center_dist"] for x in items]),
            "expected_center_dist_median": median([x["expected_center_dist"] for x in items]),
        }

    summary = {
        "dataset": str(args.dataset),
        "stages": sorted(stages),
        "rows_total": len(rows),
        "bboxes_audited": sum(len(v) for v in by_stage.values()),
        "skipped_missing_substrate": skipped_missing_substrate,
        "skipped_no_bbox": skipped_no_bbox,
        "center_threshold": args.center_threshold,
        "expected_threshold": args.expected_threshold,
        "overall": summarize([x for items in by_stage.values() for x in items]),
        "by_stage": {k: summarize(v) for k, v in sorted(by_stage.items())},
        "by_category": {k: summarize(v) for k, v in sorted(by_category.items())},
        "issue_count": len(issues),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "placement_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out / "placement_issues.jsonl", "w", encoding="utf-8") as fh:
        for item in sorted(issues, key=lambda x: (x["center_collapse"], -x["image_center_dist"]))[:500]:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    if not args.no_contact_sheet:
        draw_debug(issues, args.out / "placement_issues.jpg", max_tiles=args.max_examples)

    print(f"[placement-audit] rows={len(rows)} bboxes={summary['bboxes_audited']} "
          f"issues={summary['issue_count']}")
    print(f"[placement-audit] overall: {summary['overall']}")
    for k, v in summary["by_stage"].items():
        print(f"  - {k}: {v}")
    print(f"[placement-audit] report -> {args.out / 'placement_audit.json'}")


if __name__ == "__main__":
    main()
