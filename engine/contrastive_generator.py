"""七态对比样本矩阵生成器 (Seven-State Contrastive Generator)。
基于用户严谨的多模态评测体系，对给定的参考目标 A 与困难干扰项 B，在同一商品基底上生成 7 种互补状态：
1. CLEAN_BASE_NEG: 原始干净商品底图 -> 不存在 A (基础负样本)
2. CLEAR_POS: 正面清晰标准印花 -> 存在并完全匹配 A (定位与身份确认)
3. AFFINE_MATERIAL_POS: 透视变换与材质融合 -> 存在且匹配 A (跨视角与物理外观变化)
4. MILD_DEGRADED_POS: 轻度模糊或微小褶皱 -> 存在且保持正标签 (鲁棒性，防漏检)
5. SEVERE_INDETERMINATE: 严重模糊或大面积遮挡 -> 存在疑似痕迹但关键特征缺失 -> 证据不足 (显式不确定性建模)
6. HARD_DISTRACTOR_NEG: 添加高相似对象 B (如越替中、意替法) -> 对 A 不匹配，对 B 匹配 (细粒度困难负样本)
7. COUNTERFACTUAL_DELETED_NEG: 删除/无目标对照 -> 不存在 A (反事实干净底版)
"""
import copy
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from .coords import mask_to_rel_bbox_1000
from .reference_bank import ReferenceBank
from .compositor import CommodityCompositor


class ContrastiveGenerator:
    """七态样本生成总控"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.compositor = CommodityCompositor(width=width, height=height)

    def generate_7_state_matrix(self, ref_id: str,
                                substrate_type: str = "tshirt_fabric",
                                base_seed: int = 42) -> List[Dict[str, Any]]:
        """为指定的 ref_id (例如 'cn' 或 'fr') 生成一套完整的 7 态对照样本及密集标注。"""
        ref_meta = ReferenceBank.get_meta(ref_id)
        distractor_id = ReferenceBank.get_confusable_distractor_id(ref_id)
        distractor_meta = ReferenceBank.get_meta(distractor_id) if distractor_id else None

        # 1. 准备参考对象 A 与干扰对象 B 的标准图
        img_a = ReferenceBank.load_canonical_image(ref_id, width=320, height=213)
        img_b = ReferenceBank.load_canonical_image(distractor_id, width=320, height=213) if distractor_id else None

        # 2. 生成基底商品画布
        bg_rgb, lighting = self.compositor.generate_substrate(substrate_type=substrate_type, seed=base_seed)

        # 预设在底板上的贴附区域 (4 点透视四边形)
        # 居中偏胸口区域
        quad_normal = np.float32([
            [160, 130],
            [480, 130],
            [480, 350],
            [160, 350]
        ])

        # 带有 3D 侧倾角度与梯形透视的四边形
        quad_affine = np.float32([
            [150, 150],
            [460, 110],
            [490, 340],
            [180, 380]
        ])

        results = []

        # =====================================================================
        # 状态 1: CLEAN_BASE_NEG (干净底版基础负样本)
        # =====================================================================
        img_state_1 = self.compositor.finalize_image_symmetric(bg_rgb)
        results.append({
            "state_key": "CLEAN_BASE_NEG",
            "state_name": "原始干净商品底图",
            "image": img_state_1,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": None,
            "match_status": "absent",
            "recognizability": "none",
            "confidence": 1.0,
            "is_positive": False,
            "grounding_reason": "商品表面平整纯净，未检出任何印花、徽标或旗帜图案。"
        })

        # =====================================================================
        # 状态 2: CLEAR_POS (正面清晰正样本)
        # =====================================================================
        warped_a_flat, mask_a_flat = self.compositor.apply_homography_warp(img_a, quad_normal)
        blended_flat = self.compositor.blend_onto_substrate(bg_rgb, lighting, warped_a_flat, blend_mode="silk_screen")
        img_state_2 = self.compositor.finalize_image_symmetric(blended_flat)
        bbox_flat = mask_to_rel_bbox_1000(mask_a_flat, self.width, self.height)
        results.append({
            "state_key": "CLEAR_POS",
            "state_name": "正面清晰正样本",
            "image": img_state_2,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": bbox_flat,
            "match_status": "exact_match",
            "recognizability": "high",
            "confidence": 0.99,
            "is_positive": True,
            "grounding_reason": f"检测到商品正面印有清晰完整的 {ref_meta['name_zh']}，特征细节与标准参考图完全一致。"
        })

        # =====================================================================
        # 状态 3: AFFINE_MATERIAL_POS (透视形变与材质融合正样本)
        # =====================================================================
        warped_a_aff, mask_a_aff = self.compositor.apply_homography_warp(img_a, quad_affine)
        blended_aff = self.compositor.blend_onto_substrate(bg_rgb, lighting, warped_a_aff, blend_mode="silk_screen")
        img_state_3 = self.compositor.finalize_image_symmetric(blended_aff)
        bbox_aff = mask_to_rel_bbox_1000(mask_a_aff, self.width, self.height)
        results.append({
            "state_key": "AFFINE_MATERIAL_POS",
            "state_name": "透视形变与材质融合正样本",
            "image": img_state_3,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": bbox_aff,
            "match_status": "variant_match",
            "recognizability": "high",
            "confidence": 0.96,
            "is_positive": True,
            "grounding_reason": f"图案存在透视倾斜和织物纹理褶皱，但核心拓扑结构与关键部件可完整辨认，判定与参考图 {ref_meta['name_zh']} 匹配。"
        })

        # =====================================================================
        # 状态 4: MILD_DEGRADED_POS (轻度退化鲁棒正样本)
        # =====================================================================
        degraded_mild = self.compositor.apply_mild_degradation(blended_aff, mask_a_aff)
        img_state_4 = self.compositor.finalize_image_symmetric(degraded_mild)
        results.append({
            "state_key": "MILD_DEGRADED_POS",
            "state_name": "轻度模糊退化正样本",
            "image": img_state_4,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": bbox_aff,
            "match_status": "robust_match",
            "recognizability": "medium_high",
            "confidence": 0.90,
            "is_positive": True,
            "grounding_reason": f"图案因拍摄轻微失焦或磨损产生轻度模糊，但主要色彩搭配与标志性轮廓依然清晰可辨，保持正样本匹配。"
        })

        # =====================================================================
        # 状态 5: SEVERE_INDETERMINATE (严重退化不确定性样本)
        # =====================================================================
        degraded_severe, severe_mask = self.compositor.apply_severe_degradation(blended_aff, mask_a_aff, strategy="heavy_blur")
        img_state_5 = self.compositor.finalize_image_symmetric(degraded_severe)
        results.append({
            "state_key": "SEVERE_INDETERMINATE",
            "state_name": "重度模糊不确定性样本",
            "image": img_state_5,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": bbox_aff,
            "match_status": "insufficient_evidence",
            "recognizability": "uncertain",
            "confidence": 0.35,
            "is_positive": False,
            "grounding_reason": f"区域内虽有色块残存，但因重度散焦/遮挡导致关键构图和区分性细节完全无法辨识，证据不足，拒绝强行断言。"
        })

        # =====================================================================
        # 状态 6: HARD_DISTRACTOR_NEG (相似困难干扰负样本)
        # =====================================================================
        if img_b is not None:
            warped_b, mask_b = self.compositor.apply_homography_warp(img_b, quad_affine)
            blended_b = self.compositor.blend_onto_substrate(bg_rgb, lighting, warped_b, blend_mode="silk_screen")
            img_state_6 = self.compositor.finalize_image_symmetric(blended_b)
            bbox_b = mask_to_rel_bbox_1000(mask_b, self.width, self.height)

            distractor_reason = ref_meta.get("distractor_reason", {}).get(distractor_id, "细粒度特征不同")
            results.append({
                "state_key": "HARD_DISTRACTOR_NEG",
                "state_name": "相似对象困难负样本",
                "image": img_state_6,
                "target_ref_id": ref_id,
                "target_name": ref_meta["name_zh"],
                "distractor_ref_id": distractor_id,
                "distractor_name": distractor_meta["name_zh"],
                "bbox_1000": bbox_b,
                "match_status": "hard_mismatch_distractor",
                "recognizability": "high",
                "confidence": 0.98,
                "is_positive": False,
                "grounding_reason": f"目标区域实际为 {distractor_meta['name_zh']}，与查询参考图 {ref_meta['name_zh']} 不匹配。判定理由：{distractor_reason}"
            })

        # =====================================================================
        # 状态 7: COUNTERFACTUAL_DELETED_NEG (反事实删除负样本)
        # =====================================================================
        # 生成同款底图经过微弱局部织物修整（无目标）
        img_state_7 = self.compositor.finalize_image_symmetric(bg_rgb, quality=85)
        results.append({
            "state_key": "COUNTERFACTUAL_DELETED_NEG",
            "state_name": "反事实删除负样本",
            "image": img_state_7,
            "target_ref_id": ref_id,
            "target_name": ref_meta["name_zh"],
            "bbox_1000": None,
            "match_status": "absent",
            "recognizability": "none",
            "confidence": 1.0,
            "is_positive": False,
            "grounding_reason": f"未在商品图上发现参考目标 {ref_meta['name_zh']} 的任何残留印迹。"
        })

        return results
