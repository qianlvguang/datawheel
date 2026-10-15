"""真实服装与实物商品载体的内容与浓度阶梯合成控制引擎 (Apparel & Density Engine)。
实现无扩散模型介入下的全参数化、像素可控图像合成与密集标注。
彻底告别抽象背景，将印花与图徽无缝融合到真实电商级服装与商品模板中：
- 短袖圆领 T 恤 (含领口螺纹、尺码标、落肩线、布料悬垂感与柔和投影)
- 街头套头连帽卫衣 (含连帽抽绳、金属扣、袋鼠口袋与罗纹)
- 环保棉帆布托特包 (含挂绳织带、包底角缝线与帆布挺括质感)

控制维度一：内容 (Content Types)
1. MODULAR_CREATURE: 原创积木实体 (可控眼数、角形、胸标、配色，无先验作弊)
2. OFFICIAL_FLAG_EMBLEM: 官方标志与国徽 (西班牙海格力斯国徽、巴西天球仪、欧盟十二星环)
3. COMPOUND_BRAND_CREST: 复合图徽 (先锋盾形 + 放射星芒 + 文字飘带)

控制维度二：浓度 (Concentration & Density Levels)
1. SPARSE_CHEST_MOTIF (低空间浓度): 真实 T 恤左胸口袋位小微标 (面积占比 ~8%)
2. MEDIUM_COMPOUND (中空间浓度): 纯白 T 恤胸口黄金正位大幅印花 (面积占比 ~30%)
3. DENSE_MONOGRAM_ALLOVER (高空间老花浓度): 帆布托特包全包身错位重复老花阵列 (覆盖率 >70%)
4. WASHED_VINTAGE_FADE (稀释与磨损物理浓度): 曜石黑连帽卫衣 50% 水洗做旧透底 + 裂纹磨损
5. CLUTTERED_MULTI_DISTRACTORS (困难干扰项群体浓度): 藏青卫衣前襟同时散落目标与 3 款细粒度混淆项
"""
import math
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

from .coords import mask_to_rel_bbox_1000
from .compositor import CommodityCompositor
from .apparel_mockup import ApparelMockupFactory


class ContentAndDensityEngine:
    """真实服装与实物商品载体的内容与浓度阶梯控制引擎"""

    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height
        self.compositor = CommodityCompositor(width=width, height=height)

    @staticmethod
    def _generate_crackle_mask(width: int, height: int, crack_density: float = 0.08) -> np.ndarray:
        """生成复古丝网印花磨损与细微龟裂的二值/灰度掩模 (0.0~1.0)。"""
        noise = np.random.normal(0, 1.0, (height // 4, width // 4)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        cracks = np.clip(1.0 - (np.abs(noise) > 1.6).astype(np.float32) * 0.85, 0.15, 1.0)
        return cracks

    # =========================================================================
    # 浓度实验 1: SPARSE_CHEST_MOTIF (低空间浓度，麻灰T恤左胸口袋小微标)
    # =========================================================================
    def render_sparse_chest_motif(self, element_rgba: Image.Image,
                                  color_key: str = "heather_grey",
                                  seed: int = 101) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """在真实短袖 T 恤左胸口袋位置放置小微标 (面积占比约 8%)。"""
        base_img, garment_mask, meta = ApparelMockupFactory.render_apparel_substrate(
            "tshirt", color_key=color_key, width=self.width, height=self.height
        )

        quad = meta["zones"]["left_chest"]
        warped, mask = self.compositor.apply_homography_warp(element_rgba, quad)

        bg_rgb = np.array(base_img.convert("RGB"))
        lighting = np.ones((self.height, self.width), dtype=np.float32)
        blended = self.compositor.blend_onto_substrate(bg_rgb, lighting, warped, blend_mode="silk_screen")
        final_img = self.compositor.finalize_image_symmetric(blended)

        bbox = mask_to_rel_bbox_1000(mask, self.width, self.height)
        annotations = [{
            "label": "target_sparse_chest_motif",
            "bbox_1000": bbox,
            "density_category": "sparse_spatial_density",
            "coverage_ratio": 0.08,
            "ink_opacity": 1.0,
            "is_target": True
        }]
        return final_img, annotations, meta

    # =========================================================================
    # 浓度实验 2: MEDIUM_COMPOUND (中空间浓度，纯白T恤中心大幅印花)
    # =========================================================================
    def render_medium_compound(self, element_rgba: Image.Image,
                               color_key: str = "white",
                               seed: int = 102) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """在纯棉白 T 恤胸口中心黄金区域贴附标准中等幅面印花 (约 30% 面积占比)。"""
        base_img, garment_mask, meta = ApparelMockupFactory.render_apparel_substrate(
            "tshirt", color_key=color_key, width=self.width, height=self.height
        )

        quad = meta["zones"]["center_chest"]
        warped, mask = self.compositor.apply_homography_warp(element_rgba, quad)

        bg_rgb = np.array(base_img.convert("RGB"))
        lighting = np.ones((self.height, self.width), dtype=np.float32)
        blended = self.compositor.blend_onto_substrate(bg_rgb, lighting, warped, blend_mode="silk_screen")
        final_img = self.compositor.finalize_image_symmetric(blended)

        bbox = mask_to_rel_bbox_1000(mask, self.width, self.height)
        annotations = [{
            "label": "target_medium_compound",
            "bbox_1000": bbox,
            "density_category": "medium_spatial_density",
            "coverage_ratio": 0.30,
            "ink_opacity": 0.95,
            "is_target": True
        }]
        return final_img, annotations, meta

    # =========================================================================
    # 浓度实验 3: DENSE_MONOGRAM_ALLOVER (高空间老花浓度，帆布托特包全包身平铺)
    # =========================================================================
    def render_dense_monogram_allover(self, element_rgba: Image.Image,
                                      color_key: str = "canvas_beige",
                                      tile_size: int = 65,
                                      seed: int = 103) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """在真实环保棉帆布托特包包身上错位平铺老花阵列 (覆盖率 >70%)。"""
        base_img, garment_mask, meta = ApparelMockupFactory.render_apparel_substrate(
            "tote_bag", color_key=color_key, width=self.width, height=self.height
        )

        unit_elem = element_rgba.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        w, h = self.width, self.height

        # 托特包主体有效平铺范围 (x: 170~470, y: 245~570)
        tote_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        step_x = int(tile_size * 1.45)
        step_y = int(tile_size * 1.3)

        row_idx = 0
        for y in range(240, 560, step_y):
            offset_x = (step_x // 2) if (row_idx % 2 == 1) else 0
            for x in range(165 + offset_x, 475, step_x):
                tote_canvas.paste(unit_elem, (x, y), mask=unit_elem)
            row_idx += 1

        bg_rgb = np.array(base_img.convert("RGB"))
        tote_np = np.array(tote_canvas)
        lighting = np.ones((h, w), dtype=np.float32)

        # 调低单印饱和度，呈现棉帆布印花质感
        blended = self.compositor.blend_onto_substrate(bg_rgb, lighting, tote_np, blend_mode="silk_screen")
        final_img = self.compositor.finalize_image_symmetric(blended)

        monogram_bbox = mask_to_rel_bbox_1000(tote_np[:, :, 3], w, h)
        annotations = [{
            "label": "dense_monogram_pattern",
            "bbox_1000": monogram_bbox,
            "density_category": "dense_monogram_spatial_density",
            "coverage_ratio": 0.72,
            "is_target": True
        }]
        return final_img, annotations, meta

    # =========================================================================
    # 浓度实验 4: WASHED_VINTAGE_FADE (稀释做旧与磨损物理浓度，曜石黑连帽卫衣)
    # =========================================================================
    def render_washed_vintage_fade(self, element_rgba: Image.Image,
                                   color_key: str = "black",
                                   opacity: float = 0.50,
                                   seed: int = 104) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """在曜石黑连帽卫衣胸前呈现 50% 稀释水洗透底 + 表面龟裂磨损印花。"""
        base_img, garment_mask, meta = ApparelMockupFactory.render_apparel_substrate(
            "hoodie", color_key=color_key, width=self.width, height=self.height
        )

        quad = meta["zones"]["center_chest"]
        warped, mask = self.compositor.apply_homography_warp(element_rgba, quad)

        # 1. 降低 Alpha 通道浓度 (物理透底)
        faded_warped = warped.copy().astype(np.float32)
        faded_warped[:, :, 3] = faded_warped[:, :, 3] * opacity

        # 2. 叠加热裂纹做旧 (Crackle wear mask)
        cracks = self._generate_crackle_mask(self.width, self.height)
        faded_warped[:, :, 3] = faded_warped[:, :, 3] * cracks
        faded_warped = np.clip(faded_warped, 0, 255).astype(np.uint8)

        bg_rgb = np.array(base_img.convert("RGB"))
        lighting = np.ones((self.height, self.width), dtype=np.float32)
        blended = self.compositor.blend_onto_substrate(bg_rgb, lighting, faded_warped, blend_mode="silk_screen")
        final_img = self.compositor.finalize_image_symmetric(blended)

        bbox = mask_to_rel_bbox_1000(faded_warped[:, :, 3], self.width, self.height)
        annotations = [{
            "label": "target_washed_vintage_fade",
            "bbox_1000": bbox,
            "density_category": "washed_crackle_physical_density",
            "coverage_ratio": 0.28,
            "ink_opacity": opacity,
            "wear_level": "high_crackle_washed",
            "is_target": True
        }]
        return final_img, annotations, meta

    # =========================================================================
    # 浓度实验 5: CLUTTERED_MULTI_DISTRACTORS (困难干扰项群体密度，藏青连帽卫衣)
    # =========================================================================
    def render_cluttered_multi_distractors(self, target_rgba: Image.Image,
                                           distractor_list: List[Image.Image],
                                           color_key: str = "navy_blue",
                                           seed: int = 105) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """在藏青色连帽卫衣前襟分布 4 个槽位（目标怪兽与 3 款细粒度混淆怪兽）。"""
        base_img, garment_mask, meta = ApparelMockupFactory.render_apparel_substrate(
            "hoodie", color_key=color_key, width=self.width, height=self.height
        )

        # 卫衣前襟黄金 4 槽位
        slots = [
            np.float32([[230, 215], [315, 215], [315, 305], [230, 305]]),  # 槽位 0 (左上)
            np.float32([[325, 215], [410, 215], [410, 305], [325, 305]]),  # 槽位 1 (右上 - 设为目标)
            np.float32([[230, 315], [315, 315], [315, 405], [230, 405]]),  # 槽位 2 (左下)
            np.float32([[325, 315], [410, 315], [410, 405], [325, 405]])   # 槽位 3 (右下)
        ]

        target_slot_idx = 1  # 目标位于右上槽位
        combined_rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        annotations = []

        # 贴附目标
        warped_target, mask_target = self.compositor.apply_homography_warp(target_rgba, slots[target_slot_idx])
        combined_rgba = np.maximum(combined_rgba, warped_target)
        target_bbox = mask_to_rel_bbox_1000(mask_target, self.width, self.height)
        annotations.append({
            "label": "primary_target",
            "bbox_1000": target_bbox,
            "slot_idx": target_slot_idx,
            "is_target": True,
            "distractor_type": "none"
        })

        # 贴附干扰项
        dist_slot_indices = [idx for idx in range(len(slots)) if idx != target_slot_idx]
        for i, slot_idx in enumerate(dist_slot_indices):
            dist_img = distractor_list[i % len(distractor_list)]
            warped_dist, mask_dist = self.compositor.apply_homography_warp(dist_img, slots[slot_idx])
            combined_rgba = np.maximum(combined_rgba, warped_dist)
            dist_bbox = mask_to_rel_bbox_1000(mask_dist, self.width, self.height)
            annotations.append({
                "label": f"confusable_distractor_{i+1}",
                "bbox_1000": dist_bbox,
                "slot_idx": slot_idx,
                "is_target": False,
                "distractor_type": "confusable_candidate"
            })

        bg_rgb = np.array(base_img.convert("RGB"))
        lighting = np.ones((self.height, self.width), dtype=np.float32)
        blended = self.compositor.blend_onto_substrate(bg_rgb, lighting, combined_rgba, blend_mode="silk_screen")
        final_img = self.compositor.finalize_image_symmetric(blended)

        return final_img, annotations, meta
