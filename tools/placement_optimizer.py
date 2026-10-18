#!/usr/bin/env python3
"""M2：确定性图案摆放优化器。

输入：商品图 / 设计图 / 可印区（substrate metadata 或 --zone-rect）；
输出：最佳候选渲染图、归一化 bbox、候选评分 manifest、debug 叠加图。

流程：候选布局采样 → engine/compositor 透视/褶皱融合 → 视觉评分 → 局部精修。
不调用扩散模型；用于替代"模型把图案放画面中心"。

用法：
  .venv/bin/python tools/placement_optimizer.py \
    --image assets/commodity_substrates/real_1500/comm_studio_topwear_19457.jpg \
    --substrate-id comm_studio_topwear_19457 --zone-id center_chest_graphic \
    --design assets/official_flags/fr.svg \
    --out output/placement/audit_demo
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.compositor import CommodityCompositor
from engine.coords import mask_to_rel_bbox_1000
from placement_common import (
    find_zone_spec, load_substrates, zone_pixel_rect,
)

DEFAULT_OUT = PROJECT_ROOT / "output" / "placement"


def load_design(path: Path) -> Image.Image:
    if path.suffix.lower() == ".svg":
        import resvg_py
        svg = path.read_text(encoding="utf-8")
        png = resvg_py.svg_to_bytes(svg, width=480, dpi=96.0)
        import io
        return Image.open(io.BytesIO(png)).convert("RGBA")
    return Image.open(path).convert("RGBA")


def make_quad(cx: float, cy: float, tw: float, th: float, angle_deg: float,
              persp: Tuple[float, float, float, float, float, float, float, float]) -> np.ndarray:
    hw, hh = tw / 2.0, th / 2.0
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = [
        (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh),  # tl, tr, br, bl
    ]
    quad = []
    for i, (x, y) in enumerate(base):
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        dx = persp[i * 2]
        dy = persp[i * 2 + 1]
        quad.append([cx + rx + dx, cy + ry + dy])
    return np.float32(quad)


def sample_candidate(rng: random.Random, zone: Tuple[int, int, int, int],
                     design_w: int, design_h: int, max_fill: float) -> Dict[str, Any]:
    x0, y0, x1, y1 = zone
    zw, zh = x1 - x0, y1 - y0
    max_scale = min(zw / max(1, design_w), zh / max(1, design_h))
    fill = rng.uniform(0.55, max_fill)
    tw = max(8.0, design_w * max_scale * fill)
    th = max(8.0, design_h * max_scale * fill)
    if tw > zw or th > zh:
        ratio = min(zw / tw, zh / th)
        tw *= ratio * 0.98
        th *= ratio * 0.98
    cx = rng.uniform(x0 + tw / 2, max(x0 + tw / 2, x1 - tw / 2))
    cy = rng.uniform(y0 + th / 2, max(y0 + th / 2, y1 - th / 2))
    angle = rng.uniform(-5.0, 5.0)
    jitter = rng.uniform(0.0, 0.025) * min(zw, zh)
    persp = tuple(rng.uniform(-jitter, jitter) for _ in range(8))
    return {"cx": cx, "cy": cy, "tw": tw, "th": th, "angle": angle, "persp": persp}


def perturb_candidate(rng: random.Random, cand: Dict[str, Any],
                      zone: Tuple[int, int, int, int]) -> Dict[str, Any]:
    x0, y0, x1, y1 = zone
    zw, zh = x1 - x0, y1 - y0
    jitter = min(zw, zh) * 0.02
    out = dict(cand)
    out["cx"] = min(max(x0 + cand["tw"] / 2, cand["cx"] + rng.uniform(-jitter, jitter)),
                    max(x0 + cand["tw"] / 2, x1 - cand["tw"] / 2))
    out["cy"] = min(max(y0 + cand["th"] / 2, cand["cy"] + rng.uniform(-jitter, jitter)),
                    max(y0 + cand["th"] / 2, y1 - cand["th"] / 2))
    out["tw"] = max(8.0, cand["tw"] * rng.uniform(0.96, 1.04))
    out["th"] = max(8.0, cand["th"] * rng.uniform(0.96, 1.04))
    out["angle"] = cand["angle"] + rng.uniform(-1.5, 1.5)
    out["persp"] = tuple(v + rng.uniform(-jitter, jitter) for v in cand["persp"])
    return out


def evaluate(blended: np.ndarray, alpha: np.ndarray, bg: np.ndarray,
             zone: Tuple[int, int, int, int], design_w: int, design_h: int) -> Dict[str, Any]:
    h, w = alpha.shape
    x0, y0, x1, y1 = zone
    active = alpha > 25
    n_active = int(active.sum())
    zone_area = max(1, (x1 - x0) * (y1 - y0))
    if n_active == 0:
        return {"score": 0.0, "error": "empty_alpha"}
    zone_mask = np.zeros((h, w), dtype=bool)
    zone_mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    inside = float(zone_mask[active].mean())
    area_ratio = n_active / zone_area
    bbox = mask_to_rel_bbox_1000(alpha, w, h, threshold=25)
    if bbox:
        by0, bx0, by1, bx1 = bbox
        bcx = (bx0 + bx1) / 2.0 / 1000 * w
        bcy = (by0 + by1) / 2.0 / 1000 * h
    else:
        bcx, bcy = 0, 0
    zcx, zcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    zdiag = max(1.0, math.hypot(x1 - x0, y1 - y0))
    center_dist = math.hypot(bcx - zcx, bcy - zcy) / zdiag
    center_score = max(0.0, 1.0 - center_dist / 0.25)
    if bbox:
        bx0p = bx0 * w / 1000.0
        by0p = by0 * h / 1000.0
        bx1p = bx1 * w / 1000.0
        by1p = by1 * h / 1000.0
        clear = min(bx0p - x0, by0p - y0, x1 - bx1p, y1 - by1p)
        clear = max(0.0, clear) / max(1.0, min(x1 - x0, y1 - y0))
        edge_score = min(1.0, clear / 0.08)
        bbox_w = max(1.0, bx1p - bx0p)
        bbox_h = max(1.0, by1p - by0p)
    else:
        edge_score, bbox_w, bbox_h = 0.0, 1.0, 1.0
    diff = np.abs(blended.astype(np.int16) - bg.astype(np.int16)).mean(axis=2)
    visibility = float(diff[active].mean() / 255.0) if n_active else 0.0
    vis_score = min(1.0, visibility / 0.05)
    aspect = (bbox_w / bbox_h) if bbox_h > 0 else 1.0
    design_aspect = design_w / max(1.0, design_h)
    aspect_score = max(0.0, 1.0 - abs(math.log(max(aspect, 1e-6) / design_aspect)) / 0.7)
    if 0.08 <= area_ratio <= 0.65:
        area_score = 1.0
    else:
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.32) / 0.45)
    score = (0.35 * inside + 0.20 * area_score + 0.20 * center_score
             + 0.15 * edge_score + 0.05 * vis_score + 0.05 * aspect_score)
    return {
        "score": round(float(score), 4),
        "inside_zone": round(inside, 4),
        "area_ratio": round(area_ratio, 4),
        "center_dist": round(center_dist, 4),
        "edge_clearance": round(clear, 4) if bbox else 0.0,
        "visibility": round(visibility, 4),
        "aspect_ratio": round(aspect, 4),
        "bbox": bbox,
    }


def draw_debug(bg: np.ndarray, best_rgb: np.ndarray, zone: Tuple[int, int, int, int],
               bbox: Optional[List[int]], out_path: Path) -> None:
    img = Image.fromarray(best_rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = zone
    d.rectangle([x0, y0, x1, y1], outline=(0, 180, 255), width=3)
    d.line([img.width / 2 - 10, img.height / 2, img.width / 2 + 10, img.height / 2],
           fill=(255, 0, 0), width=2)
    d.line([img.width / 2, img.height / 2 - 10, img.width / 2, img.height / 2 + 10],
           fill=(255, 0, 0), width=2)
    if bbox:
        by0, bx0, by1, bx1 = bbox
        d.rectangle([bx0 * img.width / 1000, by0 * img.height / 1000,
                     bx1 * img.width / 1000, by1 * img.height / 1000],
                    outline=(255, 40, 40), width=3)
    img.save(out_path, quality=92)


def render_candidate(comp: CommodityCompositor, design: Image.Image, bg: np.ndarray,
                     lighting: np.ndarray, zone: Tuple[int, int, int, int],
                     cand: Dict[str, Any], seed: int, dark_surface: bool = False
                     ) -> Tuple[Dict[str, Any], np.ndarray]:
    quad = make_quad(cand["cx"], cand["cy"], cand["tw"], cand["th"],
                     cand["angle"], cand["persp"])
    warped, _ = comp.apply_homography_warp(design, quad)
    displaced, alpha = comp.apply_fold_displacement(
        warped, bg, displacement_scale=random.Random(seed).uniform(2.5, 4.5))
    blended = comp.blend_onto_substrate(
        bg, lighting, displaced,
        blend_mode="opaque_screenprint" if dark_surface else "real_garment_screenprint",
        ink_opacity=0.97 if dark_surface else 0.94,
        weave_strength=0.35 if dark_surface else 0.65)
    metrics = evaluate(blended, alpha, bg, zone, design.width, design.height)
    return metrics, blended


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic pattern placement optimizer")
    parser.add_argument("--image", type=Path, default=None, help="商品图（默认使用 substrate-id 对应文件）")
    parser.add_argument("--substrate-id", default=None, help="assets/manifests/substrate_quality.json 中的底板 ID")
    parser.add_argument("--design", type=Path, required=True, help="设计图（SVG/PNG/JPG）")
    parser.add_argument("--zone-id", default=None, help="品类模板 zone id（默认第一个）")
    parser.add_argument("--zone-rect", default=None,
                        help="自定义可印区 y0,x0,y1,x1（0~1，相对图像）；无 substrate 时使用")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "candidate")
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--refine-samples", type=int, default=24)
    parser.add_argument("--max-fill", type=float, default=0.92)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    substrates = load_substrates()
    sub = substrates.get(args.substrate_id) if args.substrate_id else None
    image_path = args.image
    if image_path is None and sub:
        image_path = PROJECT_ROOT / sub["file_path"]
    if image_path is None:
        parser.error("需要 --image 或 --substrate-id")

    img = Image.open(PROJECT_ROOT / image_path if not Path(image_path).is_absolute() else image_path).convert("RGB")
    width, height = img.size
    comp = CommodityCompositor(width, height)
    if sub:
        rgb, lighting, _meta = comp.load_real_packshot_letterboxed(
            str(PROJECT_ROOT / sub["file_path"]))
    else:
        rgb, lighting = comp.load_real_packshot(str(image_path))

    if args.zone_rect:
        y0, x0, y1, x1 = [float(v) for v in args.zone_rect.split(",")]
        zone = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
        zone_note = "custom"
    elif sub:
        spec = find_zone_spec(sub, args.zone_id)
        zone = zone_pixel_rect(sub, spec, width, height)
        zone_note = f"substrate:{sub['substrate_id']}:{spec['id']}"
    else:
        parser.error("无 substrate 时必须提供 --zone-rect")

    design = load_design(PROJECT_ROOT / args.design if not args.design.is_absolute() else args.design)
    dark_surface = bool(sub and (sub.get("dark") or sub.get("difficulty") == "hard"))
    rng = random.Random(args.seed)
    candidates: List[Dict[str, Any]] = []
    best = None
    for i in range(args.candidates):
        cand = sample_candidate(rng, zone, design.width, design.height, args.max_fill)
        metrics, rendered = render_candidate(comp, design, rgb, lighting, zone, cand,
                                             args.seed + i, dark_surface)
        item = {"candidate_id": f"c{i:04d}", "params": cand, "metrics": metrics, "rgb": rendered}
        candidates.append(item)
        if best is None or metrics["score"] > best["metrics"]["score"]:
            best = item
    # 局部精修
    for r in range(args.refine_rounds):
        base = dict(best["params"])
        for j in range(args.refine_samples):
            cand = perturb_candidate(rng, base, zone)
            metrics, rendered = render_candidate(comp, design, rgb, lighting, zone, cand,
                                                 args.seed + 1000 + r * 100 + j, dark_surface)
            item = {"candidate_id": f"r{r}_{j:03d}", "params": cand, "metrics": metrics, "rgb": rendered}
            candidates.append(item)
            if metrics["score"] > best["metrics"]["score"]:
                best = item

    args.out.mkdir(parents=True, exist_ok=True)
    out_image = args.out / "best_render.jpg"
    Image.fromarray(best["rgb"]).save(out_image, quality=93)
    draw_debug(rgb, best["rgb"], zone, best["metrics"].get("bbox"), args.out / "debug.jpg")
    candidates_sorted = sorted(candidates, key=lambda x: x["metrics"]["score"], reverse=True)
    manifest = {
        "image": str(image_path),
        "design": str(args.design),
        "zone": list(zone),
        "zone_note": zone_note,
        "candidates": len(candidates),
        "best": {
            "candidate_id": best["candidate_id"],
            "params": best["params"],
            "metrics": best["metrics"],
            "low_confidence": best["metrics"]["score"] < args.min_score,
        },
        "top3": [{"candidate_id": x["candidate_id"], "metrics": x["metrics"]}
                 for x in candidates_sorted[:3]],
    }
    (args.out / "placement_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    top_path = args.out / "candidates.jsonl"
    with open(top_path, "w", encoding="utf-8") as fh:
        for x in candidates_sorted[: max(10, args.refine_samples)]:
            fh.write(json.dumps({
                "candidate_id": x["candidate_id"],
                "params": x["params"],
                "metrics": x["metrics"],
            }, ensure_ascii=False) + "\n")
    print(f"[placement] image={image_path}")
    print(f"[placement] zone={zone} ({zone_note})")
    print(f"[placement] best={best['candidate_id']} metrics={best['metrics']}")
    if best["metrics"]["score"] < args.min_score:
        print(f"[placement] WARNING: low confidence (< {args.min_score})")
    print(f"[placement] render -> {out_image}")
    print(f"[placement] metrics -> {args.out / 'placement_metrics.json'}")


if __name__ == "__main__":
    main()
