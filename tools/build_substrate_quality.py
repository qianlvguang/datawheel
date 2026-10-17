#!/usr/bin/env python3
"""离线构建 1,500 真实商品底板的可印性与难度索引 (Substrate Printability & Difficulty Index)。

v2 的核心质量修复：
- v1 把非方形底图直接 resize 成 640x640，造成商品拉伸变形；
- v1 在花纹繁复/深色/过小/含皮肤或背景的区域贴图，印花不可辨认却仍标注可见；
- v1 会把图案印到人脸、手臂、白底等不合理区域。

流程：letterbox 保真排版 → 初始可印区 → 含皮肤/背景则向内收缩重试 →
输出稳定 zone_canvas_norm、面料表面描述、easy/medium/hard 难度分档。
产物：assets/manifests/substrate_quality.json（100% 离线）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifests" / "commodity_1500.json"
OUT_PATH = ASSETS_DIR / "manifests" / "substrate_quality.json"

CANVAS = 640
PRINT_CATEGORIES = {
    "Topwear", "Dress", "Bags", "Shoes", "Bottomwear", "Headwear", "Pants",
    "Top", "Jacket", "Coat", "Skirt", "Shirt", "Cardigan", "Sweater", "Shorts", "Vest",
}
SHRINK_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]


def letterbox_transform(src_w: int, src_h: int) -> Dict[str, float]:
    scale = min(CANVAS / float(src_w), CANVAS / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return {"scale": round(scale, 5), "new_w": new_w, "new_h": new_h,
            "ox": (CANVAS - new_w) // 2, "oy": (CANVAS - new_h) // 2}


def map_zone_to_canvas(zone_norm: List[float], tr: Dict[str, float]) -> List[float]:
    y0, x0, y1, x1 = zone_norm
    vals = [(tr["oy"] + y0 * tr["new_h"]) / CANVAS, (tr["ox"] + x0 * tr["new_w"]) / CANVAS,
            (tr["oy"] + y1 * tr["new_h"]) / CANVAS, (tr["ox"] + x1 * tr["new_w"]) / CANVAS]
    vals = [max(0.0, min(1.0, v)) for v in vals]
    return [round(v, 4) for v in vals]


def estimate_background_mask(arr: np.ndarray) -> np.ndarray:
    """检测与图像边界连通的近纯白影棚背景。"""
    h, w = arr.shape[:2]
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg_color = np.median(border, axis=0)
    if bg_color.mean() < 232:
        return np.zeros((h, w), dtype=bool)
    diff = np.abs(arr.astype(np.int16) - bg_color.astype(np.int16)).max(axis=2)
    near = (diff <= 10).astype(np.uint8)
    _, labels = cv2.connectedComponents(near, connectivity=4)
    border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)
    return np.isin(labels, list(border_labels))


def compute_zone_metrics(crop_rgb: np.ndarray, bg_crop: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr, cb, yy = ycrcb[:, :, 1], ycrcb[:, :, 2], ycrcb[:, :, 0]
    skin = (cr > 133) & (cr < 180) & (cb > 77) & (cb < 130) & (yy > 60)
    return {"luma_mean": float(gray.mean()), "luma_std": float(gray.std()),
            "grad_mean": float(np.mean(grad) / 255.0), "edge_density": float(np.mean(grad > 60)),
            "color_std": float(np.mean(lab[:, :, 1:].std(axis=(0, 1)))),
            "skin_ratio": float(np.mean(skin)), "bg_ratio": float(np.mean(bg_crop))}


def shrink_zone(zone: List[float], margin: float) -> List[float]:
    y0, x0, y1, x1 = zone
    dy, dx = (y1 - y0) * margin, (x1 - x0) * margin
    return [round(v, 4) for v in (y0 + dy, x0 + dx, y1 - dy, x1 - dx)]


def analyze_substrate(item: Dict[str, Any], arr: np.ndarray, bg_mask: np.ndarray) -> Optional[Dict[str, Any]]:
    tr = letterbox_transform(arr.shape[1], arr.shape[0])
    initial_zone = map_zone_to_canvas(item.get("printable_bbox_norm", [0.25, 0.25, 0.75, 0.75]), tr)
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for margin in SHRINK_MARGINS:
        zone = shrink_zone(initial_zone, margin)
        y0, x0, y1, x1 = zone
        px0, px1 = max(0, int(x0 * CANVAS)), min(CANVAS, int(x1 * CANVAS))
        py0, py1 = max(0, int(y0 * CANVAS)), min(CANVAS, int(y1 * CANVAS))
        if px1 - px0 < 70 or py1 - py0 < 70:
            break
        m = compute_zone_metrics(arr[py0:py1, px0:px1], bg_mask[py0:py1, px0:px1])
        area = ((px1 - px0) / CANVAS) * ((py1 - py0) / CANVAS)
        min_side = min(px1 - px0, py1 - py0)
        if area >= 0.015 and min_side >= 70:
            penalty = m["skin_ratio"] * 2.2 + m["bg_ratio"] * 1.4
            score = penalty - 0.35 * min(area / 0.09, 1.0)
            candidate = {"margin": margin, "zone": zone, "area": area, "min_side": min_side,
                         "metrics": m, "score": score}
            if best is None or score < best[0]:
                best = (score, candidate)
        if m["skin_ratio"] < 0.022 and m["bg_ratio"] < 0.06 and area >= 0.025 and min_side >= 95:
            break
    if best is None:
        return None
    chosen = best[1]
    m = chosen["metrics"]
    category = item.get("sub_category") or "Other"
    gray_all = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ycrcb_all = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr_a, cb_a, y_a = ycrcb_all[:, :, 1], ycrcb_all[:, :, 2], ycrcb_all[:, :, 0]
    skin_all = (cr_a > 133) & (cr_a < 180) & (cb_a > 77) & (cb_a < 130) & (y_a > 60)
    nonwhite = ~np.all(arr > 242, axis=2)
    global_skin_ratio = float(skin_all.sum() / max(int(nonwhite.sum()), 1))
    person_worn = global_skin_ratio >= 0.002
    geom_ok = chosen["area"] >= 0.020 and chosen["min_side"] >= 90
    content_ok = m["skin_ratio"] < 0.035 and m["bg_ratio"] < 0.10
    eligible = bool(geom_ok and content_ok and category in PRINT_CATEGORIES)
    zone = chosen["zone"]
    return {
        "substrate_id": item["substrate_id"], "source": item.get("source"),
        "sub_category": category, "category_zh": item.get("category_zh"),
        "category_en": item.get("category_en"), "file_path": item["file_path"],
        "scene_style_zh": item.get("scene_style_zh", ""), "scene_style_en": item.get("scene_style_en", ""),
        "width": int(arr.shape[1]), "height": int(arr.shape[0]), "letterbox": tr,
        "zone_canvas_norm": zone, "zone_margin": chosen["margin"],
        "zone_area_ratio": round(chosen["area"], 4), "zone_min_side_px": int(chosen["min_side"]),
        "zone_aspect": round((zone[3] - zone[1]) / max(zone[2] - zone[0], 1e-6), 3),
        "global_skin_ratio": round(global_skin_ratio, 4), "person_worn": bool(person_worn),
        "luma_mean": round(m["luma_mean"], 1), "luma_std": round(m["luma_std"], 1),
        "grad_mean": round(m["grad_mean"], 4), "edge_density": round(m["edge_density"], 3),
        "color_std": round(m["color_std"], 1), "skin_ratio": round(m["skin_ratio"], 4),
        "bg_ratio": round(m["bg_ratio"], 4), "dark": bool(m["luma_mean"] < 75),
        "studio": item.get("source") == "fashion_products_studio", "difficulty": "medium",
        "score_raw": round(m["grad_mean"] * 0.45 + m["luma_std"] / 255.0 * 0.35 + m["color_std"] / 60.0 * 0.20, 4),
        "geom_ok": bool(geom_ok), "content_ok": bool(content_ok), "print_eligible": eligible,
    }


def assign_difficulty(results: List[Dict[str, Any]]) -> None:
    eligible = [r for r in results if r["print_eligible"]]
    if not eligible:
        return
    scores = np.array([r["score_raw"] for r in eligible])
    q40, q75 = np.percentile(scores, [40, 75])
    for r in eligible:
        if r["score_raw"] > q75:
            r["difficulty"] = "hard"
        elif r["dark"]:
            r["difficulty"] = "medium"
        elif r["score_raw"] <= q40:
            r["difficulty"] = "easy"
        else:
            r["difficulty"] = "medium"


def surface_description(r: Dict[str, Any]) -> Tuple[str, str]:
    if r["grad_mean"] > 0.16 or r["luma_std"] > 62:
        return "繁复印花/强纹理面料表面", "busy patterned or heavily textured fabric surface"
    if r["color_std"] > 16:
        return "细腻图案/暗纹面料表面", "subtly patterned fabric surface"
    if r["dark"]:
        return "深色纯色面料表面", "dark solid fabric surface"
    return "自然褶皱与光影渐变的面料表面", "naturally folded fabric surface"


def contact_sheet(items: List[Dict[str, Any]], out_path: Path, n: int = 16) -> None:
    pool = [it for it in items if it["print_eligible"]]
    picks = sorted(pool, key=lambda r: r["score_raw"])[: n // 2] + sorted(pool, key=lambda r: -r["score_raw"])[: n // 2]
    tiles = []
    for it in picks:
        img = Image.open(PROJECT_ROOT / it["file_path"]).convert("RGB")
        tr = it["letterbox"]
        canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        canvas.paste(img.resize((tr["new_w"], tr["new_h"]), Image.Resampling.LANCZOS), (tr["ox"], tr["oy"]))
        draw = ImageDraw.Draw(canvas)
        y0, x0, y1, x1 = it["zone_canvas_norm"]
        draw.rectangle([x0 * CANVAS, y0 * CANVAS, x1 * CANVAS, y1 * CANVAS], outline=(255, 0, 0), width=3)
        draw.text((8, 8), f"{it['difficulty']} {it['sub_category']} skin={it['skin_ratio']:.2f}", fill=(255, 0, 0))
        tiles.append(canvas.resize((320, 320), Image.Resampling.LANCZOS))
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 320), (245, 245, 245))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 320, (i // cols) * 320))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    print(f"Contact sheet saved to {out_path}")


def main() -> None:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    print(f"Analyzing {len(items)} substrates ...")
    results: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        path = PROJECT_ROOT / item["file_path"]
        if not path.exists():
            continue
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        tr = letterbox_transform(*pil.size)
        canvas = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        canvas.paste(pil.resize((tr["new_w"], tr["new_h"]), Image.Resampling.LANCZOS), (tr["ox"], tr["oy"]))
        arr = np.asarray(canvas)
        quality = analyze_substrate(item, arr, estimate_background_mask(arr))
        if quality is not None:
            quality["surface_zh"], quality["surface_en"] = surface_description(quality)
            results.append(quality)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)} analyzed")

    assign_difficulty(results)
    eligible = [r for r in results if r["print_eligible"]]
    by_difficulty = {d: sum(1 for r in eligible if r["difficulty"] == d) for d in ("easy", "medium", "hard")}
    by_category: Dict[str, Dict[str, int]] = {}
    for r in eligible:
        bucket = by_category.setdefault(r["sub_category"], {"eligible": 0, "easy": 0})
        bucket["eligible"] += 1
        bucket["easy"] += int(r["difficulty"] == "easy")
    payload = {"version": "substrate_quality_v3", "canvas": CANVAS, "total": len(results),
               "print_eligible": len(eligible), "difficulty_counts": by_difficulty,
               "category_counts": by_category, "items": results}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"print_eligible: {len(eligible)}/{len(results)}, difficulty: {by_difficulty}")
    print("categories:", json.dumps(by_category, ensure_ascii=False))
    contact_sheet(results, PROJECT_ROOT / "output" / "_debug_substrate_contact_sheet.jpg")


if __name__ == "__main__":
    main()
