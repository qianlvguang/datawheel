#!/usr/bin/env python3
"""
Extract and curate 1,500 real commodity substrate images from cached HuggingFace Parquet shards:
1. /tmp/fashion_products_0.parquet (1,200 stratified studio packshots across 10 categories)
2. /tmp/fashionpedia_val.parquet (300 high-res apparel/streetwear images with ground-truth garment bboxes)

Computes foreground masks, safe printable placement zones, and dominant fabric color palettes,
saving images to `assets/commodity_substrates/real_1500/` and metadata to `assets/manifests/commodity_1500.json`.
"""

import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "commodity_substrates" / "real_1500"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "manifests" / "commodity_1500.json"

FP0_PATH = Path("/tmp/fashion_products_0.parquet")
FPEDIA_PATH = Path("/tmp/fashionpedia_val.parquet")

# Target quotas for studio packshots (total = 1,200)
STUDIO_QUOTAS = {
    "Topwear": 450,
    "Bags": 200,
    "Shoes": 150,
    "Bottomwear": 100,
    "Watches": 70,
    "Headwear": 60,
    "Dress": 50,
    "Wallets": 40,
    "Loungewear": 40,
    "Belts": 20,
    "Eyewear": 20,
}

CATEGORY_TRANSLATIONS = {
    "Topwear": ("上装/T恤与卫衣", "Apparel Topwear (T-Shirt / Shirt / Sweatshirt)"),
    "Bags": ("时尚箱包/双肩包与手提包", "Fashion Bag / Backpack / Tote"),
    "Shoes": ("鞋履/运动休闲鞋", "Footwear / Sneakers & Casual Shoes"),
    "Bottomwear": ("下装/休闲裤与牛仔裤", "Bottomwear / Trousers & Jeans"),
    "Watches": ("腕表/精密钟表配饰", "Wristwatch / Horology Accessory"),
    "Headwear": ("帽饰/棒球帽与针织帽", "Headwear / Cap & Beanie"),
    "Dress": ("连衣裙/一体式时装", "One-Piece Dress / Apparel"),
    "Wallets": ("皮具钱夹/卡包", "Leather Wallet / Cardholder"),
    "Loungewear": ("家居服/休闲卫衣套装", "Loungewear / Casual Comfort Apparel"),
    "Belts": ("腰带/皮革配饰", "Belt / Leather Accessory"),
    "Eyewear": ("眼镜/太阳镜配饰", "Eyewear / Sunglasses"),
}

FPEDIA_GARMENT_CATS = {
    0: ("衬衫/罩衫时装", "Editorial Shirt / Blouse"),
    1: ("T恤/卫衣上装", "Editorial Top / T-Shirt / Sweatshirt"),
    2: ("针织毛衣/羊毛衫", "Editorial Sweater / Knitwear"),
    3: ("开衫外套", "Editorial Cardigan"),
    4: ("夹克/机能外套", "Editorial Jacket / Outerwear"),
    5: ("马甲/背心", "Editorial Vest"),
    6: ("长裤/时装下装", "Editorial Pants / Trousers"),
    7: ("短裤/休闲下装", "Editorial Shorts"),
    8: ("半身裙", "Editorial Skirt"),
    9: ("长款大衣/风衣", "Editorial Coat / Trench"),
    10: ("时装连衣裙", "Editorial Dress"),
    28: ("街拍箱包/手提包", "Editorial Bag / Handbag"),
}

GENDER_ZH = {
    "Men": "男士",
    "Women": "女士",
    "Unisex": "男女同款/中性",
    "Boys": "男童/青少年",
    "Girls": "女童/青少年",
}


def classify_rgb_color(r: float, g: float, b: float) -> Tuple[str, str, str]:
    """Classify an RGB triplet into natural Chinese & English fabric color names and brightness tone."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    sat = (max_c - min_c) / (max_c + 1e-5)

    tone = "dark" if lum < 85 else ("light" if lum > 190 else "mid")

    if sat < 0.15:
        if lum < 45:
            return "深炭黑", "Charcoal Black", "dark"
        elif lum < 95:
            return "深灰/石板灰", "Dark Slate Gray", "dark"
        elif lum < 160:
            return "中性灰/麻灰", "Heather Gray", "mid"
        elif lum < 215:
            return "浅银灰/米白", "Light Silver / Off-White", "light"
        else:
            return "象牙白/纯白", "Ivory / Pure White", "light"

    # Chromatic colors via HSV hue
    rgb_arr = np.uint8([[[int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))]]])
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)[0][0]
    h = int(hsv[0])  # 0..179 in OpenCV

    if h < 8 or h >= 168:
        return ("酒红/深绯红", "Burgundy / Crimson", tone) if lum < 110 else ("正红/珊瑚红", "Crimson / Coral Red", tone)
    elif h < 22:
        if sat < 0.45 and lum > 130:
            return "卡其米色/驼色", "Khaki Beige / Camel", tone
        elif lum < 100:
            return "深棕/咖啡色", "Espresso / Dark Brown", tone
        else:
            return "活力橙/琥珀橘", "Vibrant Orange / Amber", tone
    elif h < 36:
        return ("姜黄/芥末黄", "Mustard / Ochre Yellow", tone) if lum < 150 else ("柠檬黄/明黄", "Lemon / Bright Yellow", tone)
    elif h < 78:
        return ("军绿/橄榄绿", "Olive / Forest Green", tone) if lum < 115 else ("翠绿/薄荷绿", "Emerald / Mint Green", tone)
    elif h < 102:
        return "青碧色/湖水蓝", "Teal / Cyan Blue", tone
    elif h < 132:
        return ("海军蓝/藏青色", "Navy / Midnight Blue", tone) if lum < 105 else ("皇家蓝/天蓝色", "Royal / Sky Blue", tone)
    elif h < 152:
        return ("深紫/梅子色", "Deep Plum / Violet", tone) if lum < 110 else ("薰衣草紫/丁香紫", "Lavender / Lilac Purple", tone)
    else:
        return "玫瑰粉/品红", "Rose Pink / Magenta", tone


def detect_studio_printable_zone(img_rgb: np.ndarray) -> Tuple[List[float], Tuple[str, str, str]]:
    """
    Detect non-background foreground product bounding box on studio packshot,
    returning normalized [ymin, xmin, ymax, xmax] printable zone and dominant color.
    """
    h, w = img_rgb.shape[:2]
    # Border pixels estimate background color
    border = np.concatenate([
        img_rgb[0, :, :], img_rgb[-1, :, :], img_rgb[:, 0, :], img_rgb[:, -1, :]
    ], axis=0)
    bg_color = np.median(border, axis=0)

    diff = np.linalg.norm(img_rgb.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    mask = diff > 22.0

    ys, xs = np.where(mask)
    if len(ys) < 200:
        # Fallback central box
        ymin, xmin, ymax, xmax = 0.25, 0.25, 0.72, 0.75
    else:
        y0, y1 = float(np.percentile(ys, 5)) / h, float(np.percentile(ys, 95)) / h
        x0, x1 = float(np.percentile(xs, 8)) / w, float(np.percentile(xs, 92)) / w
        # Inset to the printable central surface (chest / bag body / side panel)
        fh = max(0.15, y1 - y0)
        fw = max(0.15, x1 - x0)
        ymin = round(float(np.clip(y0 + 0.18 * fh, 0.10, 0.75)), 4)
        ymax = round(float(np.clip(y0 + 0.68 * fh, ymin + 0.15, 0.90)), 4)
        xmin = round(float(np.clip(x0 + 0.20 * fw, 0.10, 0.75)), 4)
        xmax = round(float(np.clip(x0 + 0.80 * fw, xmin + 0.15, 0.90)), 4)

    # Sample color from printable zone
    py0, py1 = int(ymin * h), max(int(ymin * h) + 2, int(ymax * h))
    px0, px1 = int(xmin * w), max(int(xmin * w) + 2, int(xmax * w))
    patch = img_rgb[py0:py1, px0:px1].reshape(-1, 3)
    med_rgb = np.median(patch, axis=0)
    color_info = classify_rgb_color(float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2]))
    return [ymin, xmin, ymax, xmax], color_info


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_items: List[Dict] = []

    print("[1/2] Reading /tmp/fashion_products_0.parquet for 1,200 studio commodity images...")
    table_fp0 = pq.read_table(
        FP0_PATH,
        columns=["id", "masterCategory", "gender", "subCategory", "image"]
    )
    rows_fp0 = table_fp0.to_pylist()

    # Group indices by subCategory
    by_subcat: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows_fp0):
        sc = r.get("subCategory") or "Other"
        by_subcat.setdefault(sc, []).append(idx)

    studio_count = 0
    for subcat, quota in STUDIO_QUOTAS.items():
        candidates = by_subcat.get(subcat, [])
        random.shuffle(candidates)
        selected = 0
        cat_zh, cat_en = CATEGORY_TRANSLATIONS[subcat]

        for idx in candidates:
            if selected >= quota:
                break
            row = rows_fp0[idx]
            img_bytes = row["image"]["bytes"]
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue

            w, h = pil_img.size
            if w < 180 or h < 180:
                continue

            img_np = np.array(pil_img)
            p_bbox, (col_zh, col_en, tone) = detect_studio_printable_zone(img_np)

            gender_raw = row.get("gender") or "Unisex"
            gender_zh = GENDER_ZH.get(gender_raw, "男女同款")

            sub_id = f"comm_studio_{subcat.lower()}_{row['id']}"
            filename = f"{sub_id}.jpg"
            out_path = OUT_DIR / filename
            pil_img.save(out_path, format="JPEG", quality=92)

            manifest_items.append({
                "substrate_id": sub_id,
                "source": "fashion_products_studio",
                "file_path": f"assets/commodity_substrates/real_1500/{filename}",
                "master_category": row.get("masterCategory", "Apparel"),
                "sub_category": subcat,
                "category_zh": f"{gender_zh}{cat_zh}",
                "category_en": f"{gender_raw}'s {cat_en}",
                "gender": gender_raw,
                "width": w,
                "height": h,
                "printable_bbox_norm": p_bbox,  # [ymin, xmin, ymax, xmax] in [0, 1]
                "dominant_color_zh": col_zh,
                "dominant_color_en": col_en,
                "fabric_tone": tone,
                "scene_style_zh": "电商标准影棚实拍白底/净底商品图",
                "scene_style_en": "Studio E-Commerce Packshot Photography",
            })
            selected += 1
            studio_count += 1

        print(f"  -> Extracted {selected}/{quota} for {subcat}")

    print(f"Total studio packshots extracted: {studio_count}")

    print("[2/2] Reading /tmp/fashionpedia_val.parquet for 300 high-res apparel images...")
    table_fpedia = pq.read_table(FPEDIA_PATH)
    rows_fpedia = table_fpedia.to_pylist()
    random.shuffle(rows_fpedia)

    target_editorial = 1500 - studio_count
    editorial_count = 0

    for row in rows_fpedia:
        if editorial_count >= target_editorial:
            break
        objs = row.get("objects", {})
        cats = objs.get("category", [])
        bboxes = objs.get("bbox", [])
        areas = objs.get("area", [])

        w = int(row["width"])
        h = int(row["height"])
        if w < 300 or h < 300:
            continue

        # Find best main garment bbox
        best_idx = -1
        best_area = 0.0
        for i, (c, bbox, area) in enumerate(zip(cats, bboxes, areas)):
            if c in FPEDIA_GARMENT_CATS and area > best_area:
                best_area = float(area)
                best_idx = i

        if best_idx < 0 or best_area < (w * h * 0.06):
            continue

        try:
            pil_img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        except Exception:
            continue

        # Resize ultra-large images so longest edge is 800px to keep disk footprint clean and fast
        orig_w, orig_h = pil_img.size
        scale = 1.0
        if max(orig_w, orig_h) > 800:
            scale = 800.0 / float(max(orig_w, orig_h))
            new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h

        # Fashionpedia bbox format is [xmin, ymin, xmax, ymax] in original pixel coords
        bx0, by0, bx1, by1 = bboxes[best_idx]
        nx0 = float(np.clip(bx0 / orig_w, 0.05, 0.85))
        ny0 = float(np.clip(by0 / orig_h, 0.05, 0.85))
        nx1 = float(np.clip(bx1 / orig_w, nx0 + 0.12, 0.95))
        ny1 = float(np.clip(by1 / orig_h, ny0 + 0.12, 0.95))

        # Inset to core printable garment surface
        gw, gh = nx1 - nx0, ny1 - ny0
        p_ymin = round(float(np.clip(ny0 + 0.18 * gh, 0.08, 0.78)), 4)
        p_ymax = round(float(np.clip(ny0 + 0.65 * gh, p_ymin + 0.14, 0.92)), 4)
        p_xmin = round(float(np.clip(nx0 + 0.20 * gw, 0.08, 0.78)), 4)
        p_xmax = round(float(np.clip(nx0 + 0.80 * gw, p_xmin + 0.14, 0.92)), 4)
        p_bbox = [p_ymin, p_xmin, p_ymax, p_xmax]

        img_np = np.array(pil_img)
        py0, py1 = int(p_ymin * new_h), max(int(p_ymin * new_h) + 2, int(p_ymax * new_h))
        px0, px1 = int(p_xmin * new_w), max(int(p_xmin * new_w) + 2, int(p_xmax * new_w))
        patch = img_np[py0:py1, px0:px1].reshape(-1, 3)
        med_rgb = np.median(patch, axis=0)
        col_zh, col_en, tone = classify_rgb_color(float(med_rgb[0]), float(med_rgb[1]), float(med_rgb[2]))

        cat_id = cats[best_idx]
        cat_zh, cat_en = FPEDIA_GARMENT_CATS[cat_id]
        img_id = row["image_id"]
        sub_id = f"comm_editorial_{cat_id}_{img_id}"
        filename = f"{sub_id}.jpg"
        out_path = OUT_DIR / filename
        pil_img.save(out_path, format="JPEG", quality=90)

        manifest_items.append({
            "substrate_id": sub_id,
            "source": "fashionpedia_editorial",
            "file_path": f"assets/commodity_substrates/real_1500/{filename}",
            "master_category": "Apparel_Editorial",
            "sub_category": cat_en.split("/")[0].replace("Editorial ", "").strip(),
            "category_zh": f"真人穿搭/秀场实拍{cat_zh}",
            "category_en": cat_en,
            "gender": "Unisex",
            "width": new_w,
            "height": new_h,
            "garment_bbox_norm": [round(ny0, 4), round(nx0, 4), round(ny1, 4), round(nx1, 4)],
            "printable_bbox_norm": p_bbox,
            "dominant_color_zh": col_zh,
            "dominant_color_en": col_en,
            "fabric_tone": tone,
            "scene_style_zh": "高分辨率街拍/时装秀场模特实景穿搭图",
            "scene_style_en": "High-Resolution Streetwear / Runway Model Apparel Photography",
        })
        editorial_count += 1

    print(f"Total editorial apparel images extracted: {editorial_count}")
    print(f"Grand total commodity substrates: {len(manifest_items)}")

    summary = {
        "dataset_name": "Datawheel Real Commodity Substrate Library 1500",
        "total_images": len(manifest_items),
        "studio_packshots_count": studio_count,
        "editorial_apparel_count": editorial_count,
        "category_breakdown": {},
        "items": manifest_items,
    }
    for item in manifest_items:
        sc = item["sub_category"]
        summary["category_breakdown"][sc] = summary["category_breakdown"].get(sc, 0) + 1

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
