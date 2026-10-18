#!/usr/bin/env python3
"""对 Datawheel v2 做小角度旋转增强（轻微"歪一点"的真实感）。

做法：
- 按 sample_id 确定性抽取一定比例样本（默认 20%）；
- 对整张图做 ±3°~±7° 旋转，再中心裁切回原尺寸（不产生白边）；
- 用 mask 精确重算所有 bbox（商品图元、参考图候选框、QA 坐标）；
- 更新 shards 的 image_sha256 / qa_pairs_json / entities_json / ref_candidates_json；
- 生成前/后对照 contact sheet 便于人工核验（默认 output/_debug/rotation_contact_sheet.jpg）。

之后需要重新执行：
  .venv/bin/python tools/normalize_dataset_language.py --skip-v6
  .venv/bin/python tools/generate_datawheel_v2.py --merge --out-dir output/datawheel_v2
  # 然后重建 V6（builders -> normalize --skip-v2 -> pack/assemble）

用法：
  .venv/bin/python tools/augment_v2_rotation.py --ratio 0.2 --dry-run
  .venv/bin/python tools/augment_v2_rotation.py --ratio 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = PROJECT_ROOT / "output" / "datawheel_v2" / "_generation" / "shard_records"
DEBUG_DIR = PROJECT_ROOT / "output" / "_debug"
BBOX_RE = re.compile(r"\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]")
TEXT_FIELDS = [
    "caption_zh", "caption_en", "visual_analysis", "knowledge_expansion",
    "qa_pairs_json", "entities_json", "ref_candidates_json",
]


def load_shards() -> List[Tuple[Path, List[Dict[str, Any]]]]:
    files = sorted(SHARDS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no shard jsonl under {SHARDS_DIR}")
    out = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out.append((path, rows))
    return out


def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """旋转 + 中心裁切回原尺寸，避免白边。"""
    w, h = img.size
    rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    left = (rotated.width - w) // 2
    top = (rotated.height - h) // 2
    return rotated.crop((left, top, left + w, top + h))


def rotate_bbox(bbox: List[int], w: int, h: int, angle: float) -> Optional[List[int]]:
    """用 mask 旋转法精确重算轴对齐 bbox：[ymin,xmin,ymax,xmax] 0~1000。"""
    y1 = max(0, min(h - 1, round(bbox[0] * h / 1000)))
    x1 = max(0, min(w - 1, round(bbox[1] * w / 1000)))
    y2 = max(y1 + 1, min(h, round(bbox[2] * h / 1000)))
    x2 = max(x1 + 1, min(w, round(bbox[3] * w / 1000)))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    left = (mask.width - w) // 2
    top = (mask.height - h) // 2
    mask = mask.crop((left, top, left + w, top + h))
    box = mask.getbbox()
    if not box:
        return None
    bx1, by1, bx2, by2 = box
    nb = [round(by1 * 1000 / h), round(bx1 * 1000 / w), round(by2 * 1000 / h), round(bx2 * 1000 / w)]
    nb = [max(0, min(1000, v)) for v in nb]
    if nb[0] >= nb[2] or nb[1] >= nb[3]:
        return None
    return nb


def iter_bbox_values(obj: Any, key_hint: str = ""):
    """遍历 JSON 中 key 含 bbox 的 4 元列表。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value) and "bbox" in key.lower():
                yield value
            else:
                yield from iter_bbox_values(value, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_bbox_values(item, key_hint)


def transform_json_bboxes(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value) and "bbox" in key.lower():
                out[key] = fn([int(v) for v in value])
            else:
                out[key] = transform_json_bboxes(value, fn)
        return out
    if isinstance(obj, list):
        return [transform_json_bboxes(item, fn) for item in obj]
    return obj


def replace_text_bboxes(text: str, mapping: Dict[Tuple[int, int, int, int], List[int]]) -> str:
    def repl(m):
        vals = tuple(int(x) for x in m.groups())
        new = mapping.get(vals)
        return f"[{new[0]}, {new[1]}, {new[2]}, {new[3]}]" if new else m.group(0)

    return BBOX_RE.sub(repl, text)


def transform_record_bboxes(row: Dict[str, Any], fn) -> Tuple[bool, Dict[Tuple[int, int, int, int], List[int]]]:
    """替换结构化 bbox，并返回 old->new 映射供文本替换。"""
    if fn is None:
        return False, {}
    mapping: Dict[Tuple[int, int, int, int], List[int]] = {}
    ok = True

    def wrap(bbox):
        nonlocal ok
        new = fn(bbox)
        if new is None:
            ok = False
            return bbox
        mapping[tuple(bbox)] = new
        return new

    try:
        boxes = json.loads(row.get("bboxes_norm_1000") or "[]")
        row["bboxes_norm_1000"] = json.dumps([wrap([int(v) for v in b]) for b in boxes], ensure_ascii=False)
        if row.get("entities_json"):
            ents = json.loads(row["entities_json"])
            ents = transform_json_bboxes(ents, wrap)
            row["entities_json"] = json.dumps(ents, ensure_ascii=False)
        if row.get("ref_candidates_json"):
            refs = json.loads(row["ref_candidates_json"])
            refs = transform_json_bboxes(refs, wrap)
            row["ref_candidates_json"] = json.dumps(refs, ensure_ascii=False)
        if row.get("qa_pairs_json"):
            qa = json.loads(row["qa_pairs_json"])
            qa = transform_json_bboxes(qa, wrap)
            row["qa_pairs_json"] = json.dumps(qa, ensure_ascii=False)
    except Exception:
        return False, {}
    if not mapping:
        return False, {}
    for field in TEXT_FIELDS:
        if isinstance(row.get(field), str):
            row[field] = replace_text_bboxes(row[field], mapping)
    return ok, mapping


def recompute_quality(row: Dict[str, Any], w: int, h: int) -> None:
    try:
        quality = json.loads(row.get("quality_json") or "{}")
    except Exception:
        return
    try:
        boxes = json.loads(row.get("bboxes_norm_1000") or "[]")
        ratios = []
        for b in boxes:
            area = max(0, (b[2] - b[0]) * (b[3] - b[1])) / 1_000_000
            ratios.append(area)
        if ratios:
            quality["bbox_area_ratio_min"] = round(min(ratios), 4)
    except Exception:
        pass
    row["quality_json"] = json.dumps(quality, ensure_ascii=False)


def choose_angle(rng: random.Random, min_angle: float, max_angle: float) -> float:
    angle = rng.uniform(min_angle, max_angle)
    return angle if rng.random() < 0.5 else -angle


def process_shards(args: argparse.Namespace) -> None:
    shards = load_shards()
    stats = Counter()
    angle_stats: List[float] = []
    contact: List[Tuple[str, Image.Image, Image.Image]] = []
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for path, rows in shards:
        for row in rows:
            sid = row.get("sample_id", "")
            if row.get("task_stage") in set(args.exclude_stages):
                stats["excluded_stage"] += 1
                continue
            meta = json.loads(row.get("render_meta_json") or "{}")
            if meta.get("rotation_deg") is not None:  # 幂等：已旋转过
                stats["already_rotated"] += 1
                continue
            rng = random.Random(f"{args.seed}|{sid}")
            if rng.random() >= args.ratio:
                stats["kept_straight"] += 1
                continue
            angle = choose_angle(rng, args.angle_min, args.angle_max)
            img_path = PROJECT_ROOT / row.get("image_path", "")
            if not img_path.exists():
                stats["missing_image"] += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                stats["read_error"] += 1
                continue
            w, h = img.size
            new_img = rotate_image(img, angle)
            old_boxes = {}
            for b in json.loads(row.get("bboxes_norm_1000") or "[]"):
                old_boxes[tuple(int(v) for v in b)] = [int(v) for v in b]
            if row.get("entities_json"):
                for b in iter_bbox_values(json.loads(row["entities_json"])):
                    old_boxes[tuple(int(v) for v in b)] = [int(v) for v in b]
            if row.get("ref_candidates_json"):
                for b in iter_bbox_values(json.loads(row["ref_candidates_json"])):
                    old_boxes[tuple(int(v) for v in b)] = [int(v) for v in b]

            def fn(b, _w=w, _h=h, _a=angle):
                return rotate_bbox(b, _w, _h, _a)

            # 无 bbox 的样本（如部分 E 遮挡样本）：只旋转图片，不改坐标
            if not old_boxes:
                if args.dry_run:
                    stats["would_rotate"] += 1
                    stats[f"stage:{row.get('task_stage','?')}"] += 1
                    angle_stats.append(angle)
                    continue
                new_img.save(img_path, format="JPEG", quality=args.quality, optimize=True)
                row["image_sha256"] = hashlib.sha256(img_path.read_bytes()).hexdigest()
                meta["rotation_deg"] = round(angle, 2)
                row["render_meta_json"] = json.dumps(meta, ensure_ascii=False)
                stats["rotated"] += 1
                stats[f"stage:{row.get('task_stage','?')}"] += 1
                angle_stats.append(angle)
                if len(contact) < 6:
                    contact.append((sid, img.copy(), new_img.copy()))
                continue

            if args.dry_run:
                new_boxes = {k: fn(k) for k in old_boxes}
                if any(v is None for v in new_boxes.values()):
                    stats["bbox_invalid_skipped"] += 1
                    continue
                stats["would_rotate"] += 1
                stats[f"stage:{row.get('task_stage','?')}"] += 1
                angle_stats.append(angle)
                continue

            ok, mapping = transform_record_bboxes(row, fn)
            if not ok or not mapping:
                stats["bbox_invalid_skipped"] += 1
                continue
            new_img.save(img_path, format="JPEG", quality=args.quality, optimize=True)
            row["image_sha256"] = hashlib.sha256(img_path.read_bytes()).hexdigest()
            meta["rotation_deg"] = round(angle, 2)
            row["render_meta_json"] = json.dumps(meta, ensure_ascii=False)
            recompute_quality(row, w, h)
            stats["rotated"] += 1
            stats[f"stage:{row.get('task_stage','?')}"] += 1
            angle_stats.append(angle)
            if len(contact) < 6:
                vis = new_img.copy()
                d = ImageDraw.Draw(vis)
                for nb in mapping.values():
                    x1 = nb[1] * w / 1000; y1 = nb[0] * h / 1000
                    x2 = nb[3] * w / 1000; y2 = nb[2] * h / 1000
                    d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
                contact.append((sid, img.copy(), vis))

    if not args.dry_run:
        for path, rows in shards:
            payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
            path.write_text(payload, encoding="utf-8")
        if contact:
            tile_w, tile_h = contact[0][1].size
            sheet = Image.new("RGB", (tile_w * 2 + 8, tile_h * len(contact) + 8 * len(contact)), (245, 245, 245))
            for i, (sid, before, after) in enumerate(contact):
                sheet.paste(before.resize((tile_w, tile_h)), (0, i * (tile_h + 8)))
                sheet.paste(after.resize((tile_w, tile_h)), (tile_w + 8, i * (tile_h + 8)))
            sheet_path = DEBUG_DIR / "rotation_contact_sheet.jpg"
            sheet.save(sheet_path, quality=90)
            print(f"[rotate] contact sheet -> {sheet_path}")

    print(f"[rotate] {'would rotate' if args.dry_run else 'rotated'} stats: {dict(stats)}")
    if angle_stats:
        print(f"[rotate] angle range: {min(angle_stats):+.2f}° ~ {max(angle_stats):+.2f}° "
              f"({len(angle_stats)} samples)")
    if args.dry_run:
        print("[rotate] dry-run: no files written")


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-angle rotation augmentation for Datawheel v2")
    parser.add_argument("--ratio", type=float, default=0.20, help="旋转样本比例（默认 0.20）")
    parser.add_argument("--angle-min", type=float, default=3.0, help="最小旋转角（默认 3°）")
    parser.add_argument("--angle-max", type=float, default=7.0, help="最大旋转角（默认 7°）")
    parser.add_argument("--seed", type=int, default=42, help="确定性抽样 seed")
    parser.add_argument("--quality", type=int, default=92, help="JPEG 保存质量")
    parser.add_argument("--exclude-stages", nargs="*", default=["stage_d_reference_match"],
                        help="不参与旋转的阶段（默认排除 stage_d_reference_match，保持候选图 pane 版式）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()
    if not (0 <= args.ratio <= 1):
        parser.error("--ratio 必须在 [0,1] 区间")
    if args.angle_min <= 0 or args.angle_max < args.angle_min:
        parser.error("角度范围非法")
    process_shards(args)


if __name__ == "__main__":
    main()
