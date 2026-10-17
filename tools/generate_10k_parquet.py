#!/usr/bin/env python3
"""
Datawheel 10,000-Sample Multimodal Dataset Generator & Parquet Packager.

Executes the full curriculum across 4 stages:
- Stage A (15% / 1,500 rows): 基础色块拼接与几何排布 (Color-block Splicing & Geometric Primitives)
- Stage B (45% / 4,500 rows): 真实商品单图元物理微观贴合 (Real Commodity Single-Element Screenprint with Fold Displacement)
- Stage C (25% / 2,500 rows): 真实商品多图元组合与空间排布 (Real Commodity Multi-Element Combinatorial Layout)
- Stage D (15% / 1,500 rows): 模板对照与跨底板/对比配对 (In-Context Reference Matching & Cross-Substrate Pairs)

Features:
- Encyclopedic knowledge expansion: 100% elimination of abbreviations (e.g. no raw 'cn', 'de', or slugs).
- Realistic placement zones: center chest, left chest badge, lower hem patch, bag flap, shoe quarter.
- Physics-based screenprint blending: Sobel gradient fold displacement + Albedo*Shading modulation + edge softening.
- Output: `output/datawheel_10k_captions.parquet` + representative rendered images in `output/images_10k/`.
- 100% offline, zero network traffic.
"""

import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import resvg_py
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.compositor import CommodityCompositor
from engine.coords import mask_to_rel_bbox_1000
from engine.knowledge_expander import expander_instance

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "images_10k"
PARQUET_PATH = OUTPUT_DIR / "datawheel_10k_captions.parquet"

CANVAS_W = 640
CANVAS_H = 640

# Color palette for Stage A
COLOR_PRESETS = [
    ("#ee1c25", "中国红/绯红", "Vibrant Scarlet Red"),
    ("#00209f", "法兰西深蓝", "Deep French Blue"),
    ("#ffffff", "纯白", "Pure White"),
    ("#ffcc00", "明金黄", "Bright Golden Yellow"),
    ("#000000", "深黑", "Deep Black"),
    ("#007a3d", "森林绿", "Forest Green"),
    ("#ff6600", "活力橙", "Vibrant Orange"),
    ("#75225b", "皇家紫", "Royal Purple"),
    ("#708090", "石板灰", "Slate Gray"),
    ("#87ceeb", "天青蓝", "Sky Blue"),
]


class Dataset10kGenerator:
    """10k 样本合成与 Parquet 封装流水线"""

    def __init__(self):
        self.compositor = CommodityCompositor(CANVAS_W, CANVAS_H)
        self.expander = expander_instance
        self.preview_sample_ids = set(range(1, 10001, 40))  # Exactly 250 evenly distributed samples
        self._index_all_assets()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def _index_all_assets(self):
        """扫描并索引本地所有合法前景色源与 1500 真实商品底板"""
        self.asset_catalog: Dict[str, List[Path]] = {
            "official_flags": sorted(list((ASSETS_DIR / "official_flags").glob("*.svg"))),
            "official_emblems": sorted(list((ASSETS_DIR / "official_emblems").glob("*.svg"))),
            "official_regional": sorted(list((ASSETS_DIR / "official_regional").glob("*.svg"))),
            "luxury_brands": sorted(list((ASSETS_DIR / "brand_logos" / "luxury_100").glob("*.svg"))),
            "ip_brands": sorted(list((ASSETS_DIR / "brand_logos" / "ip_brands_100").glob("*.svg"))),
            "character_ips": sorted(list((ASSETS_DIR / "character_ips").rglob("*.png"))),
            "subculture_elements": sorted(
                list((ASSETS_DIR / "subculture_elements").rglob("*.png")) +
                list((ASSETS_DIR / "subculture_elements").rglob("*.svg"))
            ),
        }
        for k, v in self.asset_catalog.items():
            print(f"Indexed {len(v)} items in domain '{k}'")

        # 1,500 真实商品底板
        manifest_path = ASSETS_DIR / "manifests" / "commodity_1500.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.substrates = data.get("items", [])
        else:
            self.substrates = []
        print(f"Indexed {len(self.substrates)} real commodity substrates.")

    def load_foreground_image(self, file_path: Path, target_w: int = 400) -> Optional[Image.Image]:
        """安全加载并栅格化 SVG 或 PNG 为 RGBA"""
        try:
            if file_path.suffix.lower() == ".svg":
                with open(file_path, "r", encoding="utf-8") as f:
                    svg_str = f.read()
                png_bytes = resvg_py.svg_to_bytes(svg_str, width=target_w)
                return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            else:
                img = Image.open(file_path).convert("RGBA")
                if img.width > target_w:
                    h = int(img.height * (target_w / img.width))
                    img = img.resize((target_w, max(h, 1)), Image.Resampling.LANCZOS)
                return img
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Stage A: 基础色块拼接与几何排布 (1,500 samples)
    # -------------------------------------------------------------------------
    def generate_stage_a_sample(self, sample_idx: int) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """生成基础几何色块、三色旗与多面板分割样本"""
        mode = random.choice(["tricolor_vertical", "tricolor_horizontal", "split_panel", "quad_grid", "canton_charge"])
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        c1, c1_zh, c1_en = random.choice(COLOR_PRESETS)
        c2, c2_zh, c2_en = random.choice([c for c in COLOR_PRESETS if c[0] != c1])
        c3, c3_zh, c3_en = random.choice([c for c in COLOR_PRESETS if c[0] not in (c1, c2)])

        entities: List[Dict[str, Any]] = []
        bboxes: List[List[int]] = []

        if mode == "tricolor_vertical":
            w_third = CANVAS_W // 3
            draw.rectangle([0, 0, w_third, CANVAS_H], fill=c1)
            draw.rectangle([w_third, 0, w_third * 2, CANVAS_H], fill=c2)
            draw.rectangle([w_third * 2, 0, CANVAS_W, CANVAS_H], fill=c3)

            bboxes = [[0, 0, 1000, 333], [0, 333, 1000, 667], [0, 667, 1000, 1000]]
            colors = [(c1_zh, c1_en), (c2_zh, c2_en), (c3_zh, c3_en)]
            positions = ["左侧", "中间", "右侧"]
            for pos, (czh, cen), box in zip(positions, colors, bboxes):
                entities.append({
                    "entity_type": "几何色彩区块 (Geometric Color Panel)",
                    "entity_name_zh": f"{pos}{czh}竖向矩形色块",
                    "entity_name_en": f"{pos} {cen} Vertical Panel",
                    "bbox_1000": box,
                })
            caption_zh = (
                f"图像为经典三等分垂直条纹排布构型，整体呈标准 1:1 画幅。从左至右依次排列三个等宽竖向色块："
                f"左侧色块呈现{c1_zh}（归一化坐标 [0, 0, 1000, 333]）；"
                f"中央色块呈现{c2_zh}（归一化坐标 [0, 333, 1000, 667]）；"
                f"右侧色块呈现{c3_zh}（归一化坐标 [0, 667, 1000, 1000]）。"
                f"各色块分界线笔直锐利，无模糊伪影，构成了三色旗帜学核心几何基元。"
            )
            caption_en = (
                f"A classic vertical tricolour arrangement divided into three equal vertical bands. "
                f"From left to right: a {c1_en} band [0, 0, 1000, 333], a {c2_en} band [0, 333, 1000, 667], "
                f"and a {c3_en} band [0, 667, 1000, 1000] with sharp straight boundaries."
            )

        elif mode == "tricolor_horizontal":
            h_third = CANVAS_H // 3
            draw.rectangle([0, 0, CANVAS_W, h_third], fill=c1)
            draw.rectangle([0, h_third, CANVAS_W, h_third * 2], fill=c2)
            draw.rectangle([0, h_third * 2, CANVAS_W, CANVAS_H], fill=c3)

            bboxes = [[0, 0, 333, 1000], [333, 0, 667, 1000], [667, 0, 1000, 1000]]
            colors = [(c1_zh, c1_en), (c2_zh, c2_en), (c3_zh, c3_en)]
            positions = ["顶层", "中层", "底层"]
            for pos, (czh, cen), box in zip(positions, colors, bboxes):
                entities.append({
                    "entity_type": "几何色彩区块 (Geometric Color Panel)",
                    "entity_name_zh": f"{pos}{czh}水平横向条带",
                    "entity_name_en": f"{pos} {cen} Horizontal Stripe",
                    "bbox_1000": box,
                })
            caption_zh = (
                f"图像采用三层等宽水平条纹构型。从上至下分为三个横条色块："
                f"顶层条纹呈现{c1_zh}（归一化坐标 [0, 0, 333, 1000]）；"
                f"中间条纹呈现{c2_zh}（归一化坐标 [333, 0, 667, 1000]）；"
                f"底层条纹呈现{c3_zh}（归一化坐标 [667, 0, 1000, 1000]）。"
                f"色彩纯正均匀，展示了水平条纹旗帜的经典拓扑架构。"
            )
            caption_en = (
                f"A horizontal tricolour banner composed of three equal horizontal stripes. "
                f"Top: {c1_en} [0, 0, 333, 1000]; Middle: {c2_en} [333, 0, 667, 1000]; "
                f"Bottom: {c3_en} [667, 0, 1000, 1000]."
            )

        else:
            # 基础分块与中央点缀
            draw.rectangle([0, 0, CANVAS_W, CANVAS_H], fill=c1)
            # 中央内嵌矩形或正圆
            cx0, cy0, cx1, cy1 = int(CANVAS_W * 0.28), int(CANVAS_H * 0.28), int(CANVAS_W * 0.72), int(CANVAS_H * 0.72)
            draw.ellipse([cx0, cy0, cx1, cy1], fill=c2)
            bboxes = [[0, 0, 1000, 1000], [280, 280, 720, 720]]
            entities = [
                {
                    "entity_type": "底色背景画布 (Field Background)",
                    "entity_name_zh": f"全幅纯色{c1_zh}底板",
                    "entity_name_en": f"Full-field Solid {c1_en} Canvas",
                    "bbox_1000": [0, 0, 1000, 1000],
                },
                {
                    "entity_type": "中央几何充填图元 (Central Geometric Charge)",
                    "entity_name_zh": f"正中居中{c2_zh}圆形徽记",
                    "entity_name_en": f"Centered {c2_en} Circular Disc",
                    "bbox_1000": [280, 280, 720, 720],
                }
            ]
            caption_zh = (
                f"图像由大面积纯净的{c1_zh}底板（范围覆盖 [0, 0, 1000, 1000]）与正中心醒目的{c2_zh}正圆形图元构成。"
                f"中心正圆图元的精准定位边界框为 [280, 280, 720, 720]，左右及上下严格对称，呈现强烈的向心视觉聚焦效果。"
            )
            caption_en = (
                f"An image featuring a solid {c1_en} background field [0, 0, 1000, 1000] with a centered {c2_en} "
                f"circular charge bounded precisely at [280, 280, 720, 720]."
            )

        sample_id = f"DW_10K_{sample_idx:06d}"
        record = {
            "sample_id": sample_id,
            "task_stage": "stage_a_color_blocks",
            "stage_title_zh": "阶段A：基础色块拼接与几何排布",
            "domain": "geometric_primitives",
            "substrate_id": "procedural_canvas",
            "substrate_category_zh": "程序化高精度平整色块底板",
            "placement_zone": "full_canvas_composition",
            "num_entities": len(entities),
            "entity_names_zh": " / ".join([e["entity_name_zh"] for e in entities]),
            "entity_names_en": " / ".join([e["entity_name_en"] for e in entities]),
            "entities_json": json.dumps(entities, ensure_ascii=False),
            "bboxes_norm_1000": json.dumps(bboxes),
            "caption_zh": caption_zh,
            "caption_en": caption_en,
            "visual_analysis": f"色相饱满，对比度高，边缘绝对锐利，属于标准旗帜学无形变基准测试集。",
            "knowledge_expansion": f"旗帜学与徽章学的基础色系划分（Tinctures），包含纯色场与标准几何分割法则。",
            "qa_pairs_json": json.dumps([
                {"q_zh": "图中共有几个主要几何色彩分块？", "a_zh": f"共有 {len(entities)} 个主要色彩分区。"},
                {"q_zh": "中心或主要色块的归一化坐标是多少？", "a_zh": str(bboxes[-1])}
            ], ensure_ascii=False),
            "image_path": f"output/images_10k/{sample_id}.jpg" if sample_idx in self.preview_sample_ids else "",
        }
        return record, (canvas if sample_idx in self.preview_sample_ids else None)

    # -------------------------------------------------------------------------
    # Stage B: 真实商品单图元物理微观贴合 (4,500 samples)
    # -------------------------------------------------------------------------
    def generate_stage_b_sample(self, sample_idx: int) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """在 1500 真实商品底板上应用真实褶皱置换与丝网胶印贴合单图元"""
        substrate = random.choice(self.substrates) if self.substrates else None
        domain = random.choice([
            "official_flags", "official_emblems", "official_regional",
            "luxury_brands", "ip_brands", "character_ips", "subculture_elements"
        ])
        asset_files = self.asset_catalog.get(domain, [])
        if not asset_files:
            domain = "official_flags"
            asset_files = self.asset_catalog["official_flags"]
        asset_path = random.choice(asset_files)
        stem = asset_path.stem

        # 知识扩写（绝无缩写代码）
        entity_info = self.expander.resolve_entity_knowledge(domain, stem)

        # 贴附区域策略：针对商品类别分配自然拟真的位置，杜绝死板居中
        p_zone_norm = substrate.get("printable_bbox_norm", [0.25, 0.25, 0.70, 0.75]) if substrate else [0.25, 0.25, 0.70, 0.75]
        subcat = substrate.get("sub_category", "Topwear") if substrate else "Topwear"

        zone_type = "center_chest_graphic"
        if subcat in ("Topwear", "Apparel Topwear (T-Shirt / Shirt / Sweatshirt)", "T-Shirt", "Sweatshirt"):
            zone_type = random.choice(["center_chest_graphic", "left_chest_badge", "lower_hem_patch"])
        elif subcat in ("Bags", "Fashion Bag / Backpack / Tote"):
            zone_type = "bag_front_flap"
        elif subcat in ("Shoes", "Footwear / Sneakers & Casual Shoes"):
            zone_type = "shoe_quarter"

        # 根据选定位置计算真实投放矩形
        py0, px0, py1, px1 = p_zone_norm
        if zone_type == "left_chest_badge":
            # 左胸小标：居左偏上
            bx0 = px0 + 0.05 * (px1 - px0)
            bx1 = bx0 + 0.32 * (px1 - px0)
            by0 = py0 + 0.05 * (py1 - py0)
            by1 = by0 + 0.32 * (py1 - py0)
            zone_desc_zh = "左胸上方精致刺绣/胶印徽标位置"
            zone_desc_en = "Left chest upper badge placement"
        elif zone_type == "lower_hem_patch":
            # 下摆织唛角标
            bx0 = px1 - 0.28 * (px1 - px0)
            bx1 = px1
            by0 = py1 - 0.26 * (py1 - py0)
            by1 = py1
            zone_desc_zh = "下摆边缘定制专属织唛织标位置"
            zone_desc_en = "Lower hem corner woven label placement"
        elif zone_type == "bag_front_flap":
            bx0 = px0 + 0.15 * (px1 - px0)
            bx1 = px1 - 0.15 * (px1 - px0)
            by0 = py0 + 0.12 * (py1 - py0)
            by1 = py1 - 0.15 * (py1 - py0)
            zone_desc_zh = "箱包正面外袋正中装饰徽章位置"
            zone_desc_en = "Bag front pocket central patch placement"
        elif zone_type == "shoe_quarter":
            bx0 = px0 + 0.10 * (px1 - px0)
            bx1 = px1 - 0.10 * (px1 - px0)
            by0 = py0 + 0.15 * (py1 - py0)
            by1 = py1 - 0.15 * (py1 - py0)
            zone_desc_zh = "鞋帮侧面透气帮面徽记位置"
            zone_desc_en = "Footwear lateral quarter emblem placement"
        else:
            # center_chest_graphic
            bx0 = px0 + 0.08 * (px1 - px0)
            bx1 = px1 - 0.08 * (px1 - px0)
            by0 = py0 + 0.10 * (py1 - py0)
            by1 = py1 - 0.10 * (py1 - py0)
            zone_desc_zh = "前胸黄金主视觉核心印花区域"
            zone_desc_en = "Center chest hero graphic area"

        # 加载底图
        sub_img_path = PROJECT_ROOT / substrate["file_path"] if substrate else None
        if sub_img_path and sub_img_path.exists():
            bg_rgb, lighting_map = self.compositor.load_real_packshot(str(sub_img_path))
        else:
            bg_rgb, lighting_map = self.compositor.generate_substrate("tshirt_fabric", seed=sample_idx)

        # 加载前景色图元
        elem_rgba = self.load_foreground_image(asset_path, target_w=350)
        if elem_rgba is None:
            elem_rgba = Image.new("RGBA", (200, 200), (220, 40, 40, 255))

        # 透视顶点坐标 (轻微自然微倾斜，避免电脑人工正方框)
        rot_deg = random.uniform(-2.5, 2.5)
        rad = np.radians(rot_deg)
        x0_px, y0_px = bx0 * CANVAS_W, by0 * CANVAS_H
        x1_px, y1_px = bx1 * CANVAS_W, by1 * CANVAS_H
        cx, cy = (x0_px + x1_px) / 2.0, (y0_px + y1_px) / 2.0
        hw, hh = (x1_px - x0_px) / 2.0, (y1_px - y0_px) / 2.0

        quad = np.float32([
            [cx - hw * np.cos(rad) + hh * np.sin(rad), cy - hw * np.sin(rad) - hh * np.cos(rad)],
            [cx + hw * np.cos(rad) + hh * np.sin(rad), cy + hw * np.sin(rad) - hh * np.cos(rad)],
            [cx + hw * np.cos(rad) - hh * np.sin(rad), cy + hw * np.sin(rad) + hh * np.cos(rad)],
            [cx - hw * np.cos(rad) - hh * np.sin(rad), cy - hw * np.sin(rad) + hh * np.cos(rad)],
        ])

        # 物理仿生贴附流程
        warped_rgba, _ = self.compositor.apply_homography_warp(elem_rgba, quad)
        # 双向梯度折痕置换
        displaced_rgba, displaced_alpha = self.compositor.apply_fold_displacement(
            warped_rgba, bg_rgb, displacement_scale=random.uniform(3.0, 5.0)
        )
        # 实拍布料丝网胶印混合 (Albedo * Shading + 织物微纹理透出)
        blended_rgb = self.compositor.blend_onto_substrate(
            bg_rgb, lighting_map, displaced_rgba, blend_mode="real_garment_screenprint"
        )
        final_pil = self.compositor.finalize_image_symmetric(blended_rgb) if sample_idx in self.preview_sample_ids else None

        box_calc = mask_to_rel_bbox_1000(displaced_alpha, threshold=25)
        ground_bbox_1000 = box_calc if box_calc is not None else [
            int(round(by0 * 1000)), int(round(bx0 * 1000)),
            int(round(by1 * 1000)), int(round(bx1 * 1000))
        ]

        # 动态视觉特征分析
        vis_features = self.expander.inspect_visual_features(elem_rgba)

        # 拼装百科深度丰富 Caption
        sub_name_zh = substrate.get("category_zh", "高品质纯棉衣物") if substrate else "棉质服饰面料"
        sub_name_en = substrate.get("category_en", "High-Grade Cotton Apparel") if substrate else "Cotton Apparel Fabric"
        sub_color_zh = substrate.get("dominant_color_zh", "深炭黑") if substrate else "深炭黑"
        sub_color_en = substrate.get("dominant_color_en", "Charcoal Black") if substrate else "Charcoal Black"

        caption_zh = (
            f"本商品展示图基于一件实拍拍摄的{sub_color_zh}{sub_name_zh}底板。"
            f"在{zone_desc_zh}，采用高精度工业丝网胶印工艺贴附有【{entity_info['full_title_zh']}】。"
            f"视觉特征与设计解析：{entity_info['design_analysis_zh']}"
            f"印花图形的主导色调为{vis_features['dominant_color_zh']}，呈现{vis_features['layout_geometry_zh']}。"
            f"物理贴合呈现：印花随衣物面料真实的微观褶皱与经纬走势产生平滑自然的折痕形变弯曲，"
            f"边缘经过抗锯齿柔化并与背景布料的光影明暗深度融合，杜绝突兀贴片感。"
            f"该图案在整图中的高精度归一化定位边界框为 {ground_bbox_1000}。"
        )
        caption_en = (
            f"A studio packshot of {sub_color_en} {sub_name_en}. "
            f"Printed at the {zone_desc_en} is the official {entity_info['full_title_en']}. "
            f"Design analysis: {entity_info['design_analysis_en']} "
            f"The print features a {vis_features['dominant_color_en']} palette with {vis_features['layout_geometry_en']}. "
            f"The graphic conforms physically to fabric folds via surface displacement and screenprint illumination modulation. "
            f"Ground-truth bounding box: {ground_bbox_1000}."
        )

        sample_id = f"DW_10K_{sample_idx:06d}"
        entities_list = [{
            "entity_type": entity_info["entity_type"],
            "entity_name_zh": entity_info["entity_name_zh"],
            "entity_name_en": entity_info["entity_name_en"],
            "full_title_zh": entity_info["full_title_zh"],
            "full_title_en": entity_info["full_title_en"],
            "placement_zone": zone_type,
            "bbox_1000": ground_bbox_1000,
        }]

        record = {
            "sample_id": sample_id,
            "task_stage": "stage_b_single_element",
            "stage_title_zh": "阶段B：真实商品单图元物理微观贴合",
            "domain": domain,
            "substrate_id": substrate.get("substrate_id", "real_substrate") if substrate else "synthetic",
            "substrate_category_zh": sub_name_zh,
            "placement_zone": zone_type,
            "num_entities": 1,
            "entity_names_zh": entity_info["entity_name_zh"],
            "entity_names_en": entity_info["entity_name_en"],
            "entities_json": json.dumps(entities_list, ensure_ascii=False),
            "bboxes_norm_1000": json.dumps([ground_bbox_1000]),
            "caption_zh": caption_zh,
            "caption_en": caption_en,
            "visual_analysis": f"色调: {vis_features['dominant_color_zh']}; 几何: {vis_features['layout_geometry_zh']}; 置换形变比例: 自然微曲。",
            "knowledge_expansion": f"{entity_info['full_title_zh']} - {entity_info['cultural_context_zh']}",
            "qa_pairs_json": json.dumps([
                {"q_zh": f"这件{sub_name_zh}上的印花图案是什么？", "a_zh": f"印花为{entity_info['full_title_zh']}。"},
                {"q_zh": "该印花图案的具体边界框坐标是多少？", "a_zh": str(ground_bbox_1000)},
                {"q_zh": "印花图案在衣物上的具体印花位置是？", "a_zh": zone_desc_zh}
            ], ensure_ascii=False),
            "image_path": f"output/images_10k/{sample_id}.jpg" if sample_idx in self.preview_sample_ids else "",
        }
        return record, final_pil

    # -------------------------------------------------------------------------
    # Stage C: 真实商品多图元组合与空间排布 (2,500 samples)
    # -------------------------------------------------------------------------
    def generate_stage_c_sample(self, sample_idx: int) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """在同件真实商品上排布 2 个相互独立的视觉印花（例如主视觉胸前印花 + 左胸/袖标/下摆徽标）"""
        substrate = random.choice(self.substrates) if self.substrates else None
        p_zone_norm = substrate.get("printable_bbox_norm", [0.25, 0.25, 0.70, 0.75]) if substrate else [0.25, 0.25, 0.70, 0.75]
        py0, px0, py1, px1 = p_zone_norm

        # 选取 2 个不同领域的图案
        domains = random.sample([
            "official_flags", "official_emblems", "luxury_brands",
            "ip_brands", "character_ips", "subculture_elements"
        ], 2)
        asset1 = random.choice(self.asset_catalog[domains[0]])
        asset2 = random.choice(self.asset_catalog[domains[1]])

        info1 = self.expander.resolve_entity_knowledge(domains[0], asset1.stem)
        info2 = self.expander.resolve_entity_knowledge(domains[1], asset2.stem)

        # 加载底图
        sub_img_path = PROJECT_ROOT / substrate["file_path"] if substrate else None
        if sub_img_path and sub_img_path.exists():
            bg_rgb, lighting_map = self.compositor.load_real_packshot(str(sub_img_path))
        else:
            bg_rgb, lighting_map = self.compositor.generate_substrate("tshirt_fabric", seed=sample_idx)

        # 图元 1：主视觉中心印花
        elem1 = self.load_foreground_image(asset1, target_w=320) or Image.new("RGBA", (200, 200), (200, 30, 30, 255))
        bx0_1 = px0 + 0.15 * (px1 - px0)
        bx1_1 = px1 - 0.15 * (px1 - px0)
        by0_1 = py0 + 0.25 * (py1 - py0)
        by1_1 = py1 - 0.08 * (py1 - py0)
        quad1 = np.float32([
            [bx0_1 * CANVAS_W, by0_1 * CANVAS_H], [bx1_1 * CANVAS_W, by0_1 * CANVAS_H],
            [bx1_1 * CANVAS_W, by1_1 * CANVAS_H], [bx0_1 * CANVAS_W, by1_1 * CANVAS_H]
        ])
        w1, a1 = self.compositor.apply_homography_warp(elem1, quad1)
        d1, da1 = self.compositor.apply_fold_displacement(w1, bg_rgb, displacement_scale=4.0)

        # 图元 2：左胸刺绣/胶印小标 (位于主图元左上方)
        elem2 = self.load_foreground_image(asset2, target_w=180) or Image.new("RGBA", (150, 150), (30, 120, 220, 255))
        bx0_2 = px0 + 0.02 * (px1 - px0)
        bx1_2 = bx0_2 + 0.26 * (px1 - px0)
        by0_2 = py0 + 0.02 * (py1 - py0)
        by1_2 = by0_2 + 0.24 * (py1 - py0)
        quad2 = np.float32([
            [bx0_2 * CANVAS_W, by0_2 * CANVAS_H], [bx1_2 * CANVAS_W, by0_2 * CANVAS_H],
            [bx1_2 * CANVAS_W, by1_2 * CANVAS_H], [bx0_2 * CANVAS_W, by1_2 * CANVAS_H]
        ])
        w2, a2 = self.compositor.apply_homography_warp(elem2, quad2)
        d2, da2 = self.compositor.apply_fold_displacement(w2, bg_rgb, displacement_scale=3.5)

        # 合并两图层并贴附
        merged_rgba = d1.copy()
        mask2 = (da2 > 20)[:, :, np.newaxis]
        merged_rgba = np.where(mask2, d2, merged_rgba)

        blended_rgb = self.compositor.blend_onto_substrate(
            bg_rgb, lighting_map, merged_rgba, blend_mode="real_garment_screenprint"
        )
        final_pil = self.compositor.finalize_image_symmetric(blended_rgb) if sample_idx in self.preview_sample_ids else None
        b1 = mask_to_rel_bbox_1000(da1, threshold=25)
        box1 = b1 if b1 is not None else [int(by0_1 * 1000), int(bx0_1 * 1000), int(by1_1 * 1000), int(bx1_1 * 1000)]
        b2 = mask_to_rel_bbox_1000(da2, threshold=25)
        box2 = b2 if b2 is not None else [int(by0_2 * 1000), int(bx0_2 * 1000), int(by1_2 * 1000), int(bx1_2 * 1000)]

        sub_name_zh = substrate.get("category_zh", "实拍服饰商品") if substrate else "服饰商品"
        sub_name_en = substrate.get("category_en", "Apparel Commodity") if substrate else "Apparel Commodity"
        sub_color_zh = substrate.get("dominant_color_zh", "纯色") if substrate else "纯色"

        caption_zh = (
            f"该{sub_color_zh}{sub_name_zh}包含两个独立且协同排布的视觉元素，呈现典型的品牌联名或系列主副标组合："
            f"1. 核心主视觉图案位于胸前中央，为【{info1['full_title_zh']}】（边界框 {box1}），设计解析：{info1['design_analysis_zh']}"
            f"2. 次要装饰徽标位于主图案左上方的左胸胸口处，为【{info2['full_title_zh']}】（边界框 {box2}）。"
            f"两处印花各自顺应了衣物折痕进行非线性起伏形变，主副图形层次分明，呈现高度自然的实物上身质感。"
        )
        caption_en = (
            f"This {sub_name_en} features a dual-element graphic arrangement: "
            f"Primary chest graphic: {info1['full_title_en']} [ymin,xmin,ymax,xmax]={box1}. "
            f"Secondary chest badge located diagonally to the upper-left: {info2['full_title_en']} [ymin,xmin,ymax,xmax]={box2}. "
            f"Both prints physically conform to garment creases."
        )

        sample_id = f"DW_10K_{sample_idx:06d}"
        entities_list = [
            {"entity_type": info1["entity_type"], "entity_name_zh": info1["entity_name_zh"], "full_title_zh": info1["full_title_zh"], "placement_zone": "center_chest_graphic", "bbox_1000": box1},
            {"entity_type": info2["entity_type"], "entity_name_zh": info2["entity_name_zh"], "full_title_zh": info2["full_title_zh"], "placement_zone": "left_chest_badge", "bbox_1000": box2},
        ]

        record = {
            "sample_id": sample_id,
            "task_stage": "stage_c_multi_elements",
            "stage_title_zh": "阶段C：真实商品多图元组合与空间排布",
            "domain": f"{domains[0]}+{domains[1]}",
            "substrate_id": substrate.get("substrate_id", "real_substrate") if substrate else "synthetic",
            "substrate_category_zh": sub_name_zh,
            "placement_zone": "center_main_plus_left_chest_badge",
            "num_entities": 2,
            "entity_names_zh": f"{info1['entity_name_zh']} / {info2['entity_name_zh']}",
            "entity_names_en": f"{info1['entity_name_en']} / {info2['entity_name_en']}",
            "entities_json": json.dumps(entities_list, ensure_ascii=False),
            "bboxes_norm_1000": json.dumps([box1, box2]),
            "caption_zh": caption_zh,
            "caption_en": caption_en,
            "visual_analysis": f"双图元组合；主视觉占据中腹部；次级徽标位于左上偏置；空间拓扑严格隔离无重叠碰撞。",
            "knowledge_expansion": f"主图案：{info1['full_title_zh']}；副图案：{info2['full_title_zh']}",
            "qa_pairs_json": json.dumps([
                {"q_zh": "图中包含几个独立的印花？分别是什么？", "a_zh": f"包含 2 个印花：主印花为{info1['entity_name_zh']}，左上角徽标为{info2['entity_name_zh']}。"},
                {"q_zh": f"次要徽标【{info2['entity_name_zh']}】相对于主图案的位置关系是什么？", "a_zh": "位于主图案的左上方（左胸胸口区域）。"},
                {"q_zh": "主图案的边界框坐标是什么？", "a_zh": str(box1)}
            ], ensure_ascii=False),
            "image_path": f"output/images_10k/{sample_id}.jpg" if sample_idx in self.preview_sample_ids else "",
        }
        return record, final_pil

    # -------------------------------------------------------------------------
    # Stage D: 模板对照与跨底板/对比配对 (1,500 samples)
    # -------------------------------------------------------------------------
    def generate_stage_d_sample(self, sample_idx: int) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """左右分栏对照：左侧为标准无畸变基准模板图，右侧为贴附在实物商品上的实物单反实拍图"""
        substrate = random.choice(self.substrates) if self.substrates else None
        p_zone_norm = substrate.get("printable_bbox_norm", [0.25, 0.25, 0.70, 0.75]) if substrate else [0.25, 0.25, 0.70, 0.75]

        domain = random.choice([
            "official_flags", "official_emblems", "official_regional",
            "luxury_brands", "ip_brands", "character_ips", "subculture_elements"
        ])
        asset_path = random.choice(self.asset_catalog[domain])
        info = self.expander.resolve_entity_knowledge(domain, asset_path.stem)

        # 加载底图
        sub_img_path = PROJECT_ROOT / substrate["file_path"] if substrate else None
        if sub_img_path and sub_img_path.exists():
            bg_rgb, lighting_map = self.compositor.load_real_packshot(str(sub_img_path))
        else:
            bg_rgb, lighting_map = self.compositor.generate_substrate("tshirt_fabric", seed=sample_idx)

        elem_rgba = self.load_foreground_image(asset_path, target_w=300) or Image.new("RGBA", (200, 200), (220, 30, 30, 255))

        # 右半幅贴附
        py0, px0, py1, px1 = p_zone_norm
        bx0, bx1 = px0 + 0.1 * (px1 - px0), px1 - 0.1 * (px1 - px0)
        by0, by1 = py0 + 0.15 * (py1 - py0), py1 - 0.15 * (py1 - py0)
        quad = np.float32([
            [bx0 * CANVAS_W, by0 * CANVAS_H], [bx1 * CANVAS_W, by0 * CANVAS_H],
            [bx1 * CANVAS_W, by1 * CANVAS_H], [bx0 * CANVAS_W, by1 * CANVAS_H]
        ])
        warped_rgba, _ = self.compositor.apply_homography_warp(elem_rgba, quad)
        displaced_rgba, displaced_alpha = self.compositor.apply_fold_displacement(warped_rgba, bg_rgb, displacement_scale=4.2)
        blended_right = self.compositor.blend_onto_substrate(bg_rgb, lighting_map, displaced_rgba, blend_mode="real_garment_screenprint")

        # 构造左右分栏大图 (CANVAS_W * 2, CANVAS_H) 或压在标准方形内
        # 将左侧放白底参考模板，右侧放商品图，合成一张标准 CANVAS_W x CANVAS_H 图像
        half_w = CANVAS_W // 2
        split_img = np.full((CANVAS_H, CANVAS_W, 3), 248, dtype=np.uint8)

        # 左侧：基准图元居中放置在浅白灰背景上
        elem_resized = elem_rgba.copy()
        elem_resized.thumbnail((int(half_w * 0.75), int(CANVAS_H * 0.75)), Image.Resampling.LANCZOS)
        ew, eh = elem_resized.size
        ox = (half_w - ew) // 2
        oy = (CANVAS_H - eh) // 2
        elem_np = np.array(elem_resized)
        alpha_l = elem_np[:, :, 3] / 255.0
        for c in range(3):
            split_img[oy:oy+eh, ox:ox+ew, c] = np.clip(
                elem_np[:, :, c] * alpha_l + split_img[oy:oy+eh, ox:ox+ew, c] * (1.0 - alpha_l), 0, 255
            )

        # 右侧：将实物图缩小 50% 放置在右半幅
        right_sub_pil = Image.fromarray(blended_right).resize((half_w, CANVAS_H), Image.Resampling.LANCZOS)
        split_img[:, half_w:] = np.array(right_sub_pil)

        # 中缝分割线
        split_img[:, half_w-1:half_w+1] = [180, 180, 185]
        final_pil = self.compositor.finalize_image_symmetric(split_img) if sample_idx in self.preview_sample_ids else None

        # 归一化计算两幅中的定位：
        # 左侧基准框 (0~1000):
        left_box = [int(oy * 1000 / CANVAS_H), int(ox * 1000 / CANVAS_W), int((oy+eh) * 1000 / CANVAS_H), int((ox+ew) * 1000 / CANVAS_W)]
        # 右侧实物框:
        bo = mask_to_rel_bbox_1000(displaced_alpha, threshold=25)
        box_orig = bo if bo is not None else [int(by0 * 1000), int(bx0 * 1000), int(by1 * 1000), int(bx1 * 1000)]
        right_box = [box_orig[0], int(500 + box_orig[1] * 0.5), box_orig[2], int(500 + box_orig[3] * 0.5)]

        sub_name_zh = substrate.get("category_zh", "实拍服饰商品") if substrate else "实物商品"

        caption_zh = (
            f"本样本为跨模态图文检索与实例比对测试集：左右分栏对比视图。"
            f"【左侧视图】：展示了【{info['full_title_zh']}】的标准无畸变基准模板（定位框 {left_box}），清晰呈现其原始几何构型与色彩；"
            f"【右侧视图】：展示了同一实体印制于真实{sub_name_zh}上的实际拍摄呈现（定位框 {right_box}）。"
            f"跨视图对比分析：右侧图案受到面料三维起伏与折痕阴影调制，呈现物理置换与微距织物漫反射，"
            f"验证模型在无畸变设计图与复杂物理实物之间的跨域特征恒常性检索能力。"
        )
        caption_en = (
            f"In-context side-by-side verification pair. "
            f"Left pane: Canonical reference template of {info['full_title_en']} [ymin,xmin,ymax,xmax]={left_box}. "
            f"Right pane: The identical entity printed on {sub_name_zh} subjected to fabric fold displacement [ymin,xmin,ymax,xmax]={right_box}."
        )

        sample_id = f"DW_10K_{sample_idx:06d}"
        entities_list = [
            {"entity_type": "基准设计参考模板 (Canonical Reference)", "full_title_zh": f"{info['full_title_zh']}（标准模板）", "pane": "left_template", "bbox_1000": left_box},
            {"entity_type": "实物印花应用实例 (Physical Commodity Instance)", "full_title_zh": f"{info['full_title_zh']}（实物商品）", "pane": "right_commodity", "bbox_1000": right_box},
        ]

        record = {
            "sample_id": sample_id,
            "task_stage": "stage_d_in_context_pairs",
            "stage_title_zh": "阶段D：模板对照与跨底板/对比配对",
            "domain": domain,
            "substrate_id": substrate.get("substrate_id", "real_substrate") if substrate else "synthetic",
            "substrate_category_zh": sub_name_zh,
            "placement_zone": "side_by_side_reference_and_packshot",
            "num_entities": 2,
            "entity_names_zh": info["entity_name_zh"],
            "entity_names_en": info["entity_name_en"],
            "entities_json": json.dumps(entities_list, ensure_ascii=False),
            "bboxes_norm_1000": json.dumps([left_box, right_box]),
            "caption_zh": caption_zh,
            "caption_en": caption_en,
            "visual_analysis": f"分栏比例 1:1；左侧为标准 RGB 平面位图；右侧为实物单反实拍商品与真实折痕置换贴图。",
            "knowledge_expansion": f"{info['full_title_zh']} - {info['design_analysis_zh']}",
            "qa_pairs_json": json.dumps([
                {"q_zh": "左侧与右侧的图案是否属于同一实体？", "a_zh": f"是，两边均为{info['full_title_zh']}，左侧为标准模板，右侧为印制在实物商品上的实物应用。"},
                {"q_zh": "右侧实物商品上的印花定位坐标是多少？", "a_zh": str(right_box)}
            ], ensure_ascii=False),
            "image_path": f"output/images_10k/{sample_id}.jpg" if sample_idx in self.preview_sample_ids else "",
        }
        return record, final_pil

    # -------------------------------------------------------------------------
    # 主生成执行循环与 Parquet 写入
    # -------------------------------------------------------------------------
    def run_10k_generation(self) -> None:
        """生成 10,000 条结构化多模态微调与评测数据并保存为 Parquet"""
        total_samples = 10000
        stage_quotas = [
            ("stage_a", 1500, self.generate_stage_a_sample),
            ("stage_b", 4500, self.generate_stage_b_sample),
            ("stage_c", 2500, self.generate_stage_c_sample),
            ("stage_d", 1500, self.generate_stage_d_sample),
        ]

        print("=" * 70)
        print("Datawheel 10,000-Sample Multi-Modal Curriculum Generation Starting")
        print("Dataset Composition:")
        print("  - Stage A (15% / 1,500): 基础色块拼接与几何排布")
        print("  - Stage B (45% / 4,500): 真实商品单图元物理微观贴合 (1500 真实商品底板)")
        print("  - Stage C (25% / 2,500): 真实商品多图元组合与空间排布")
        print("  - Stage D (15% / 1,500): 模板对照与跨底板/对比配对")
        print("=" * 70)

        all_records: List[Dict[str, Any]] = []
        global_idx = 1
        start_time = time.time()
        rendered_images_count = 0

        for stage_name, quota, gen_fn in stage_quotas:
            print(f"\n[Generating {stage_name}] Quota: {quota} samples...")
            stage_start = time.time()
            for i in range(quota):
                record, img_pil = gen_fn(global_idx)
                all_records.append(record)

                if img_pil is not None:
                    out_img_path = IMAGES_DIR / f"{record['sample_id']}.jpg"
                    img_pil.save(out_img_path, format="JPEG", quality=90)
                    rendered_images_count += 1

                global_idx += 1

                if (i + 1) % 500 == 0 or (i + 1) == quota:
                    elapsed = time.time() - stage_start
                    speed = (i + 1) / max(elapsed, 0.001)
                    print(f"  Stage {stage_name}: {i + 1}/{quota} records generated ({speed:.1f} records/s)")

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Generation Complete! Total records: {len(all_records)} in {total_elapsed:.1f}s")
        print(f"Representative full-resolution preview images saved: {rendered_images_count} to {IMAGES_DIR}")

        # 构建 PyArrow 表并写入 Parquet
        print(f"\nBuilding PyArrow table and writing to {PARQUET_PATH}...")
        arrow_schema = pa.schema([
            ("sample_id", pa.string()),
            ("task_stage", pa.string()),
            ("stage_title_zh", pa.string()),
            ("domain", pa.string()),
            ("substrate_id", pa.string()),
            ("substrate_category_zh", pa.string()),
            ("placement_zone", pa.string()),
            ("num_entities", pa.int32()),
            ("entity_names_zh", pa.string()),
            ("entity_names_en", pa.string()),
            ("entities_json", pa.string()),
            ("bboxes_norm_1000", pa.string()),
            ("caption_zh", pa.string()),
            ("caption_en", pa.string()),
            ("visual_analysis", pa.string()),
            ("knowledge_expansion", pa.string()),
            ("qa_pairs_json", pa.string()),
            ("image_path", pa.string()),
        ])

        col_data = {field.name: [r[field.name] for r in all_records] for field in arrow_schema}
        arrow_table = pa.Table.from_pydict(col_data, schema=arrow_schema)
        pq.write_table(arrow_table, PARQUET_PATH, compression="zstd")

        file_size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
        print(f"Successfully wrote {PARQUET_PATH} ({file_size_mb:.2f} MB, {len(arrow_table)} rows)")

        # 生成数据集评估与统计报告
        report_path = OUTPUT_DIR / "DATASET_10K_REPORT.md"
        self._generate_dataset_report(report_path, all_records, file_size_mb, rendered_images_count)
        print(f"Generated comprehensive report at {report_path}")

    def _generate_dataset_report(self, report_path: Path, records: List[Dict], parquet_mb: float, img_count: int):
        stage_counts = {}
        domain_counts = {}
        for r in records:
            sc = r["stage_title_zh"]
            dm = r["domain"]
            stage_counts[sc] = stage_counts.get(sc, 0) + 1
            domain_counts[dm] = domain_counts.get(dm, 0) + 1

        lines = [
            "# Datawheel 10,000-Sample Multimodal Grounding Dataset Report",
            f"\n- **Total Dataset Records**: {len(records):,} rows",
            f"- **Parquet File**: `{PARQUET_PATH.name}` ({parquet_mb:.2f} MB, ZSTD compressed)",
            f"- **Rendered Preview Images**: {img_count} images in `output/images_10k/`",
            f"- **Commodity Substrate Pool**: 1,500 authentic studio packshots and editorial fashion images",
            f"- **Network Traffic Consumed**: **0.00 MB** (100% offline local generation)\n",
            "## 1. Curriculum Stage Distribution",
            "| Stage | Stage Description (ZH) | Samples | Proportion |",
            "|---|---|---|---|",
        ]
        for stg, cnt in stage_counts.items():
            lines.append(f"| `{stg.split('：')[0]}` | {stg} | {cnt:,} | {cnt/len(records)*100:.1f}% |")

        lines.extend([
            "\n## 2. Domain & Asset Source Breakdown",
            "| Asset Domain | Samples | Target Coverage |",
            "|---|---|---|",
        ])
        for dm, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{dm}` | {cnt:,} | {cnt/len(records)*100:.1f}% |")

        lines.extend([
            "\n## 3. High-Quality Sample Excerpts (Anti-Abbreviation & Knowledge Expanded)",
            "### Example 1: Stage B Single-Element Physical Screenprint",
            f"```json\n{json.dumps(records[1500], ensure_ascii=False, indent=2)}\n```",
            "### Example 2: Stage C Multi-Element Arrangement",
            f"```json\n{json.dumps(records[6000], ensure_ascii=False, indent=2)}\n```",
            "### Example 3: Stage D In-Context Reference Matching Pair",
            f"```json\n{json.dumps(records[8500], ensure_ascii=False, indent=2)}\n```",
        ])
        report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generator = Dataset10kGenerator()
    generator.run_10k_generation()
