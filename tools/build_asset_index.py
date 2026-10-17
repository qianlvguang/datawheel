#!/usr/bin/env python3
"""离线构建全量资产视觉索引 (Asset Visual Index Builder)。

为 Datawheel v2 生成两类可复用缓存，避免每条样本重复渲染/计算：
1. 每个前景图元的中性视觉属性：主导色、宽高比、几何构型、dHash 感知哈希；
2. 同域视觉近邻表 (hard-negative pool)：用于参考图匹配任务的困难负样本与
   负样本/干扰样本选择。

产物：assets/manifests/asset_visual_index.json
- 100% 离线，不消耗网络流量。
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

import resvg_py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUT_PATH = ASSETS_DIR / "manifests" / "asset_visual_index.json"

THUMB = 96
DHASH_EDGE = 8

DOMAIN_GLOBS: Dict[str, List[str]] = {
    "official_flags": ["official_flags/*.svg"],
    "official_emblems": ["official_emblems/*.svg"],
    "official_regional": ["official_regional/*.svg"],
    "luxury_brands": ["brand_logos/luxury_100/*.svg"],
    "ip_brands": ["brand_logos/ip_brands_100/*.svg"],
    "character_ips": ["character_ips/**/*.png"],
    "subculture_elements": ["subculture_elements/**/*.png", "subculture_elements/**/*.svg"],
}


def domain_assets(domain: str) -> List[Path]:
    files: List[Path] = []
    for pattern in DOMAIN_GLOBS[domain]:
        files.extend(ASSETS_DIR.glob(pattern))
    return sorted({p for p in files if p.is_file() and not p.name.startswith(".")})


def load_rgba(path: Path, max_side: int = THUMB) -> Optional[Image.Image]:
    """渲染 SVG / 加载 PNG 为缩略 RGBA；失败返回 None。"""
    try:
        if path.suffix.lower() == ".svg":
            svg_str = path.read_text(encoding="utf-8")
            # 某些 SVG 含外部引用/异常属性，resvg 可能抛异常
            png_bytes = resvg_py.svg_to_bytes(svg_str, width=max_side, dpi=96.0)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        else:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def rgb_to_color_name(r: float, g: float, b: float) -> tuple[str, str]:
    """与 knowledge_expander 同口径的中文/英文色名映射（避免依赖单例的随机采样）。"""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c, min_c = max(r, g, b), min(r, g, b)
    sat = (max_c - min_c) / (max_c + 1e-5)
    if sat < 0.14:
        if lum < 50:
            return "浓郁曜石黑", "Obsidian Black"
        if lum < 120:
            return "深沉石板灰", "Slate Gray"
        if lum < 200:
            return "质感中性灰", "Neutral Heather Gray"
        return "纯净明亮白", "Pure Crisp White"
    if r > g * 1.35 and r > b * 1.35:
        return ("典雅酒红/深绯", "Deep Burgundy Red") if lum < 110 else ("鲜明中国红/赤焰红", "Vibrant Scarlet Red")
    if g > r * 1.25 and g > b * 1.25:
        return ("沉稳墨绿/橄榄绿", "Forest Olive Green") if lum < 120 else ("清爽翠绿/薄荷绿", "Vivid Emerald Green")
    if b > r * 1.25 and b > g * 1.1:
        return ("深邃海军蓝/藏青", "Deep Navy Blue") if lum < 115 else ("明媚皇家蓝/天青", "Vibrant Royal Blue")
    if r > 160 and g > 130 and b < 90:
        return "辉煌金黄/琥珀黄", "Golden Amber Yellow"
    if r > 160 and g > 75 and b < 70:
        return "活力热烈橙", "Vivid Energetic Orange"
    if r > 120 and b > 120 and g < 100:
        return "神秘魅惑紫", "Mystic Royal Purple"
    if r > 160 and b > 130 and g > 110:
        return "浪漫樱花粉", "Soft Blossom Pink"
    return "复合混色调", "Multi-Color Composite"


def geometry_from_aspect(aspect: float) -> tuple[str, str]:
    if aspect >= 1.6:
        return "横向宽幅扁平矩形构型", "Horizontal Widescreen Layout"
    if aspect <= 0.65:
        return "纵向立式纤细长条构型", "Tall Vertical Banner Proportion"
    if 0.90 <= aspect <= 1.10:
        return "正方形/对称正圆同心圆构型", "Square / Concentric Circular Geometry"
    return "标准 4:3 / 3:2 传统旗帜与徽章比例", "Classic 3:2 / 4:3 Aspect Ratio"


def dhash64(img_rgba: Image.Image) -> str:
    """感知哈希：白底合成后灰度 9x8 相邻比较，返回 16 位十六进制。"""
    white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    white.alpha_composite(img_rgba)
    gray = white.convert("L").resize((DHASH_EDGE + 1, DHASH_EDGE), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def analyze_asset(path: Path) -> Optional[Dict[str, Any]]:
    img = load_rgba(path)
    if img is None:
        return None
    arr = np.asarray(img)
    alpha = arr[:, :, 3]
    active = arr[alpha > 40]
    if len(active) == 0:
        return None
    rgb = active[:, :3].astype(np.float32)
    if len(rgb) > 4000:
        idx = np.linspace(0, len(rgb) - 1, 4000).astype(np.int64)
        rgb = rgb[idx]
    med = np.median(rgb, axis=0)
    color_zh, color_en = rgb_to_color_name(float(med[0]), float(med[1]), float(med[2]))
    w, h = img.size
    aspect = round(float(w) / max(h, 1), 3)
    geom_zh, geom_en = geometry_from_aspect(aspect)
    # 饱和度与明度统计（用于控制台/暗底适配）
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    sat = float(np.mean((mx - mn) / (mx + 1e-5)))
    return {
        "aspect_ratio": aspect,
        "geometry_zh": geom_zh,
        "geometry_en": geom_en,
        "dominant_color_zh": color_zh,
        "dominant_color_en": color_en,
        "mean_saturation": round(sat, 3),
        "alpha_ratio": round(float(np.mean(alpha > 40)), 3),
        "dhash": dhash64(img),
    }


def main() -> None:
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    neighbors: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[str] = []
    total = 0

    for domain in DOMAIN_GLOBS:
        files = domain_assets(domain)
        domain_index: Dict[str, Dict[str, Any]] = {}
        for path in files:
            info = analyze_asset(path)
            if info is None:
                failures.append(f"{domain}/{path.name}")
                continue
            domain_index[path.stem] = info
        index[domain] = domain_index

        # 同域 dHash 近邻（困难负样本池）
        stems = list(domain_index.keys())
        for stem in stems:
            h = domain_index[stem]["dhash"]
            ranked = sorted(
                ({"stem": other, "distance": hamming(h, domain_index[other]["dhash"])}
                 for other in stems if other != stem),
                key=lambda item: item["distance"],
            )[:12]
            neighbors[f"{domain}/{stem}"] = ranked
        total += len(domain_index)
        print(f"[{domain}] indexed {len(domain_index)}/{len(files)} assets")

    payload = {
        "version": "asset_visual_index_v1",
        "thumbnail_side": THUMB,
        "total_assets": total,
        "domains": index,
        "neighbors": neighbors,
        "failures": failures,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"Total indexed: {total}, failures: {len(failures)}")
    if failures:
        print("Failed assets sample:", failures[:10])


if __name__ == "__main__":
    main()
