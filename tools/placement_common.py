#!/usr/bin/env python3
"""Placement 共享常量：商品品类模板、可印区、坐标换算。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSTRATE_QUALITY = PROJECT_ROOT / "assets" / "manifests" / "substrate_quality.json"

# 与 tools/generate_datawheel_v2.py 的 ZONE_SPECS 保持一致；修改请同步两处
ZONE_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "Topwear": [
        {"id": "center_chest_graphic", "rect": (0.12, 0.18, 0.82, 0.82),
         "zh": "前胸中央主视觉印花区", "en": "center chest hero graphic area", "kind": "screenprint"},
        {"id": "left_chest_badge", "rect": (0.05, 0.04, 0.32, 0.32),
         "zh": "左胸上方小型徽标区", "en": "left chest badge area", "kind": "screenprint"},
        {"id": "lower_hem_patch", "rect": (0.72, 0.30, 0.94, 0.74),
         "zh": "下摆边缘织唛/小标区", "en": "lower hem label patch area", "kind": "woven_label"},
    ],
    "Bags": [
        {"id": "bag_front_flap", "rect": (0.12, 0.12, 0.58, 0.88),
         "zh": "包盖/正面外袋装饰徽标区", "en": "bag front flap emblem area", "kind": "heat_transfer"},
        {"id": "bag_front_panel", "rect": (0.30, 0.16, 0.82, 0.86),
         "zh": "包身正面主视觉印制区", "en": "bag front panel graphic area", "kind": "heat_transfer"},
    ],
    "Shoes": [
        {"id": "shoe_lateral_quarter", "rect": (0.18, 0.12, 0.68, 0.88),
         "zh": "鞋面/鞋帮主标识区", "en": "shoe upper emblem area", "kind": "heat_transfer"},
        {"id": "shoe_tongue_panel", "rect": (0.06, 0.26, 0.40, 0.74),
         "zh": "鞋舌/鞋面小标区", "en": "shoe tongue small-label area", "kind": "woven_label"},
    ],
    "Headwear": [
        {"id": "hat_front_panel", "rect": (0.20, 0.18, 0.72, 0.82),
         "zh": "帽冠正面主视觉区", "en": "hat front panel area", "kind": "embroidery"},
    ],
    "Bottomwear": [
        {"id": "thigh_side_graphic", "rect": (0.22, 0.16, 0.74, 0.84),
         "zh": "裤腿正面/侧面印花区", "en": "pants leg graphic area", "kind": "screenprint"},
        {"id": "pants_hem_patch", "rect": (0.74, 0.18, 0.94, 0.80),
         "zh": "裤脚口织唛/小标区", "en": "pants hem label area", "kind": "woven_label"},
    ],
    "Dress": [
        {"id": "bodice_center_graphic", "rect": (0.16, 0.16, 0.62, 0.84),
         "zh": "裙身上半部主视觉印花区", "en": "bodice center graphic area", "kind": "screenprint"},
        {"id": "skirt_hem_patch", "rect": (0.72, 0.22, 0.94, 0.78),
         "zh": "裙摆边缘下摆小标区", "en": "skirt hem label area", "kind": "woven_label"},
    ],
}
CATEGORY_ALIAS = {
    "Top": "Topwear", "Shirt": "Topwear", "Jacket": "Topwear", "Coat": "Topwear",
    "Cardigan": "Topwear", "Sweater": "Topwear", "Vest": "Topwear",
    "Pants": "Bottomwear", "Shorts": "Bottomwear", "Skirt": "Bottomwear",
    "Topwear": "Topwear", "Bottomwear": "Bottomwear", "Dress": "Dress",
    "Bags": "Bags", "Shoes": "Shoes", "Headwear": "Headwear",
}


def load_substrates(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    data = json.loads(Path(path or SUBSTRATE_QUALITY).read_text(encoding="utf-8"))
    return {item["substrate_id"]: item for item in data.get("items", [])}


def category_for(sub: Dict[str, Any]) -> str:
    return CATEGORY_ALIAS.get(sub.get("sub_category", ""), "Topwear")


def zone_specs_for(sub: Dict[str, Any]) -> List[Dict[str, Any]]:
    return ZONE_SPECS.get(category_for(sub), ZONE_SPECS["Topwear"])


def find_zone_spec(sub: Dict[str, Any], zone_id: Optional[str] = None) -> Dict[str, Any]:
    specs = zone_specs_for(sub)
    if zone_id:
        for spec in specs:
            if spec["id"] == zone_id:
                return spec
        raise KeyError(f"zone_id={zone_id} 不在 {category_for(sub)} 的模板中: {[s['id'] for s in specs]}")
    return specs[0]


def zone_pixel_rect(sub: Dict[str, Any], spec: Dict[str, Any],
                    width: int = 640, height: int = 640,
                    canvas: int = 640) -> Tuple[int, int, int, int]:
    """返回可印区内、spec 指定子区域的像素框 (x0,y0,x1,y1)。

    zone_canvas_norm 定义在 640 letterbox 画布上。目标图可能是 640 画布
    （v2 图像）或原始商品图（如 384×512）；按目标图尺寸动态计算 letterbox，
    把 640 画布坐标映射回目标图像坐标。
    """
    y0, x0, y1, x1 = sub["zone_canvas_norm"]
    ry0, rx0, ry1, rx1 = spec["rect"]
    cx0 = (x0 + rx0 * (x1 - x0)) * canvas
    cx1 = (x0 + rx1 * (x1 - x0)) * canvas
    cy0 = (y0 + ry0 * (y1 - y0)) * canvas
    cy1 = (y0 + ry1 * (y1 - y0)) * canvas
    # 目标图放进 640 画布时的动态 letterbox
    scale = min(canvas / max(1, width), canvas / max(1, height))
    new_w = width * scale
    new_h = height * scale
    ox = (canvas - new_w) / 2.0
    oy = (canvas - new_h) / 2.0
    px0 = int((cx0 - ox) / scale)
    px1 = int((cx1 - ox) / scale)
    py0 = int((cy0 - oy) / scale)
    py1 = int((cy1 - oy) / scale)
    px0, py0 = max(1, px0), max(1, py0)
    px1, py1 = min(width - 2, px1), min(height - 2, py1)
    if px1 <= px0 + 2 or py1 <= py0 + 2:
        raise ValueError(f"可印区过小: {sub['substrate_id']} {spec['id']} -> {(px0, py0, px1, py1)}")
    return px0, py0, px1, py1

def canvas_zone_pixel_rect(sub: Dict[str, Any], width: int = 640, height: int = 640) -> Tuple[int, int, int, int]:
    y0, x0, y1, x1 = sub["zone_canvas_norm"]
    return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = rect
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def bbox_to_pixels(bbox: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
    """[ymin,xmin,ymax,xmax] 0~1000 -> (x0,y0,x1,y1)。"""
    y0, x0, y1, x1 = bbox
    return (int(x0 * width / 1000), int(y0 * height / 1000),
            int(x1 * width / 1000), int(y1 * height / 1000))
