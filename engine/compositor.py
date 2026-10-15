"""物理仿真材质与仿射融合引擎 (Commodity Compositor Engine)。
核心能力：
1. 程序化合成多种商品材质底板 (纯棉布料、褶皱卫衣、马克杯曲面、牛皮纸盒)。
2. 基于透视单应性变换 (Homography) 与法线置换的 3D 贴附渲染。
3. 像素级 Alpha Mask 实时追踪：从形变后的可见区域计算精确 0~1000 相对 BBox，杜绝理论矩形偏差。
4. 物理融合 (材质乘算、边缘抗锯齿与织物微噪点)。
5. 对称管线处理：正样本、困难负样本与干净底图经历相同的传感器噪声与 JPEG 压缩，消除伪影捷径。
"""
import io
import math
import random
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

from .coords import mask_to_rel_bbox_1000, to_rel_bbox_1000


class CommodityCompositor:
    """商品仿真合成器"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    # -------------------------------------------------------------------------
    # 1. 程序化商品底板合成 (Procedural Substrate Synthesis)
    # -------------------------------------------------------------------------
    def generate_substrate(self, substrate_type: str = "tshirt_fabric",
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """生成商品底板 RGB 图像以及光照/褶皱法线强度图 (lighting_map: 0.0~1.0)。"""
        if seed is not None:
            np.random.seed(seed)

        w, h = self.width, self.height

        if substrate_type == "tshirt_fabric":
            # 基础浅灰色/燕麦色棉质 T 恤
            base_color = np.array([230, 228, 225], dtype=np.float32)
            # 经纬交错微观纺织纹理 (Weave texture)
            x = np.arange(w)
            y = np.arange(h)
            xx, yy = np.meshgrid(x, y)
            weave = 0.03 * (np.sin(xx * 1.5) + np.cos(yy * 1.5))

            # 真实衣物大尺度褶皱阴影 (Wrinkle waves)
            wrinkles = 0.12 * np.sin(xx * 0.015 + np.cos(yy * 0.02) * 2.0)
            wrinkles += 0.08 * np.sin(yy * 0.025 + xx * 0.01)

            lighting_map = np.clip(1.0 + weave + wrinkles, 0.7, 1.25)
            rgb = np.clip(base_color * lighting_map[:, :, np.newaxis], 0, 255).astype(np.uint8)

        elif substrate_type == "ceramic_mug":
            # 马克杯曲面柱状高光与阴影
            base_color = np.array([245, 245, 247], dtype=np.float32)
            x = np.linspace(-1.0, 1.0, w)
            # 柱面漫反射余弦光照 + 侧方环境遮挡
            cylinder_light = np.clip(np.cos(x * 1.4), 0.45, 1.0)
            cylinder_light = np.tile(cylinder_light, (h, 1))

            # 顶部微弱反光渐变
            y = np.linspace(0.95, 1.05, h)[:, np.newaxis]
            lighting_map = np.clip(cylinder_light * y, 0.4, 1.2)
            rgb = np.clip(base_color * lighting_map[:, :, np.newaxis], 0, 255).astype(np.uint8)

        elif substrate_type == "kraft_paper":
            # 环保牛皮纸包装箱
            base_color = np.array([198, 166, 126], dtype=np.float32)
            # 纸浆纤维粗糙噪点
            noise = np.random.normal(0, 4.0, (h, w))
            # 纸盒折痕压线
            crease = np.zeros((h, w), dtype=np.float32)
            crease_x = int(w * 0.35)
            crease[:, max(0, crease_x - 3):min(w, crease_x + 3)] = -0.15

            lighting_map = np.clip(1.0 + (noise / 255.0) + crease, 0.75, 1.2)
            rgb = np.clip(base_color * lighting_map[:, :, np.newaxis], 0, 255).astype(np.uint8)

        else:
            # 默认中性底板
            rgb = np.full((h, w, 3), 220, dtype=np.uint8)
            lighting_map = np.ones((h, w), dtype=np.float32)

        return rgb, lighting_map

    # -------------------------------------------------------------------------
    # 2. 单应性透视贴附与 Mask 提取 (Perspective Homography & Mask Tracking)
    # -------------------------------------------------------------------------
    def apply_homography_warp(self, element_rgba: Image.Image,
                              quad_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将元素贴图根据目标 4 点透视变换投射到全尺寸底板画布上。
        返回：
            warped_rgba: (H, W, 4) uint8 数组
            alpha_mask: (H, W) uint8 数组 (0 或 255)
        """
        elem_w, elem_h = element_rgba.size
        src_pts = np.float32([[0, 0], [elem_w, 0], [elem_w, elem_h], [0, elem_h]])
        dst_pts = np.float32(quad_dst)

        # 计算单应性矩阵 H
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 对 RGBA 图像进行透视变换
        elem_np = np.array(element_rgba)
        warped_rgba = cv2.warpPerspective(elem_np, H, (self.width, self.height),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))

        alpha_mask = warped_rgba[:, :, 3]
        return warped_rgba, alpha_mask

    # -------------------------------------------------------------------------
    # 3. 物理贴附融合 (Physical Material Blending)
    # -------------------------------------------------------------------------
    def blend_onto_substrate(self, bg_rgb: np.ndarray, lighting_map: np.ndarray,
                             warped_rgba: np.ndarray,
                             blend_mode: str = "silk_screen") -> np.ndarray:
        """根据材质特性进行光照调制与图层混合。
        silk_screen: 丝网印刷（保留印花固有色，但受底板光照阴影和粗糙度调制）
        """
        result = bg_rgb.copy().astype(np.float32)
        fg_rgb = warped_rgba[:, :, :3].astype(np.float32)
        alpha = (warped_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        if blend_mode == "silk_screen":
            # 印花油墨受表面光照图 (lighting_map) 乘算
            light_3d = lighting_map[:, :, np.newaxis]
            modulated_fg = np.clip(fg_rgb * (0.85 + 0.15 * light_3d), 0, 255)
            # Alpha 正常融合
            result = modulated_fg * alpha + result * (1.0 - alpha)
        else:
            result = fg_rgb * alpha + result * (1.0 - alpha)

        return np.clip(result, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 4. 可控退化算子 (Controlled Degradation Operators)
    # -------------------------------------------------------------------------
    def apply_mild_degradation(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """轻度退化：轻微高斯模糊 (sigma=1.2) + 模拟衣物局部褶皱小阴影。仍可清晰辨认身份。"""
        pil_img = Image.fromarray(img_rgb)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))
        res = np.array(blurred)
        # 轻微对比度微调
        return res

    def apply_severe_degradation(self, img_rgb: np.ndarray, mask: np.ndarray,
                                 strategy: str = "heavy_blur") -> Tuple[np.ndarray, np.ndarray]:
        """重度退化：严重模糊 (sigma=9.0) 或大面积遮挡 (>65%)。
        关键视觉特征无法被证实，标签转入 `insufficient_evidence`。
        """
        h, w = img_rgb.shape[:2]
        res = img_rgb.copy()
        remaining_mask = mask.copy()

        if strategy == "heavy_blur":
            # 剧烈失焦模糊
            pil_img = Image.fromarray(img_rgb)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=8.5))
            res = np.array(blurred)
            # 严重模糊下，虽然像素仍在，但特征有效信噪比丧失

        elif strategy == "heavy_occlusion":
            # 几何遮挡 (例如手提袋带子或重叠物体)
            active_y, active_x = np.where(mask > 20)
            if len(active_y) > 0:
                min_x, max_x = int(np.min(active_x)), int(np.max(active_x))
                min_y, max_y = int(np.min(active_y)), int(np.max(active_y))
                # 遮挡超过 70% 的面积
                occ_w = int((max_x - min_x) * 0.75)
                occ_color = np.array([45, 45, 50], dtype=np.uint8)
                res[min_y:max_y, min_x:min_x + occ_w] = occ_color
                remaining_mask[min_y:max_y, min_x:min_x + occ_w] = 0

        return res, remaining_mask

    # -------------------------------------------------------------------------
    # 5. 对称相机传感器与压缩处理 (Symmetric Post-Processing)
    # -------------------------------------------------------------------------
    def finalize_image_symmetric(self, img_rgb: np.ndarray, quality: int = 88) -> Image.Image:
        """所有样本（正负样本、退化样本、空底图）必须经历完全相同的保存与压缩管线，
        杜绝让模型通过边缘微弱锯齿或重采样伪影作弊。
        """
        # 添加极微弱传感器模拟噪点 (Sensor Noise, sigma=1.0)
        noise = np.random.normal(0, 1.0, img_rgb.shape).astype(np.float32)
        final_array = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(final_array)
        # 内存中模拟真实 JPEG 编码压缩
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
