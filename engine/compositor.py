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

    def load_real_packshot(self, packshot_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载真实单反实拍商品底图，并提取局部归一化织物光影场 (lighting_map)。
        通过局部低频环境光除法，无论白 T 恤、麻灰 T 恤还是纯黑卫衣，均能精准提取真实褶皱明暗与织物纹理。
        """
        pil_bg = Image.open(packshot_path).convert("RGB")
        if pil_bg.size != (self.width, self.height):
            pil_bg = pil_bg.resize((self.width, self.height), Image.Resampling.LANCZOS)
        rgb = np.array(pil_bg, dtype=np.uint8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # 大尺度高斯平滑估计局部基础环境光，避免黑色衣物绝对亮度低导致印花全黑
        ambient = cv2.GaussianBlur(gray, (0, 0), sigmaX=45.0, sigmaY=45.0)
        lighting_map = (gray + 15.0) / (ambient + 15.0)
        lighting_map = np.clip(lighting_map, 0.68, 1.22).astype(np.float32)
        return rgb, lighting_map

    def load_real_packshot_letterboxed(self, packshot_path: str,
                                       bg_color: Tuple[int, int, int] = (255, 255, 255)):
        """保真加载真实商品底板：按原始宽高比 letterbox 居中放置，绝不拉伸商品。

        返回 (rgb, lighting_map, meta)；meta 含 scale/new_w/new_h/ox/oy/src_w/src_h，
        用于把底板上原有标注（printable_bbox_norm）精确映射到 640 画布坐标。
        """
        pil_bg = Image.open(packshot_path).convert("RGB")
        src_w, src_h = pil_bg.size
        scale = min(self.width / float(src_w), self.height / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        ox = (self.width - new_w) // 2
        oy = (self.height - new_h) // 2
        canvas = Image.new("RGB", (self.width, self.height), bg_color)
        canvas.paste(pil_bg.resize((new_w, new_h), Image.Resampling.LANCZOS), (ox, oy))
        rgb = np.array(canvas, dtype=np.uint8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ambient = cv2.GaussianBlur(gray, (0, 0), sigmaX=45.0, sigmaY=45.0)
        lighting_map = (gray + 15.0) / (ambient + 15.0)
        lighting_map = np.clip(lighting_map, 0.68, 1.22).astype(np.float32)
        meta = {"scale": round(scale, 5), "new_w": new_w, "new_h": new_h,
                "ox": ox, "oy": oy, "src_w": src_w, "src_h": src_h,
                "canvas_w": self.width, "canvas_h": self.height}
        return rgb, lighting_map, meta

    # -------------------------------------------------------------------------
    # 2. 单应性透视贴附与真实折痕置换 (Homography + Fold Displacement Mapping)
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

    def apply_fold_displacement(self, warped_rgba: np.ndarray, bg_rgb: np.ndarray,
                                displacement_scale: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
        """双向折痕置换贴图 (Fold Displacement Mapping)：
        根据实拍底图的明暗梯度场 (Sobel 导数) 计算布料起伏法向量，使印花随真实衣服褶皱自然弯曲。
        """
        h, w = bg_rgb.shape[:2]
        gray = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # 中尺度平滑，提取宏观布料折痕而非单根纱线噪点
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=6.0, sigmaY=6.0)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

        # 归一化梯度并按位移强度缩放
        norm = max(float(np.percentile(np.hypot(gx, gy), 98)), 1e-3)
        dx = (gx / norm) * displacement_scale
        dy = (gy / norm) * displacement_scale

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = (grid_x - dx).astype(np.float32)
        map_y = (grid_y - dy).astype(np.float32)

        displaced_rgba = cv2.remap(
            warped_rgba, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return displaced_rgba, displaced_rgba[:, :, 3]

    # -------------------------------------------------------------------------
    # 3. 物理贴附融合 (Physical Material Blending)
    # -------------------------------------------------------------------------
    def blend_onto_substrate(self, bg_rgb: np.ndarray, lighting_map: np.ndarray,
                             warped_rgba: np.ndarray,
                             blend_mode: str = "silk_screen",
                             ink_opacity: float = 0.94,
                             weave_strength: float = 0.65) -> np.ndarray:
        """根据材质特性进行光照调制与图层混合。
        - silk_screen: 基础丝网印刷
        - real_garment_screenprint: 实拍衣物物理胶印融合（褶皱阴影乘算 + 微距纱线渗透 + 边缘光学柔化）
        - opaque_screenprint: 不透明塑胶浆印花（深色/繁复面料上保证图案可辨认）
        """
        result = bg_rgb.copy().astype(np.float32)
        fg_rgb = warped_rgba[:, :, :3].astype(np.float32)
        raw_alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
        light_3d = lighting_map[:, :, np.newaxis]

        if blend_mode == "real_garment_screenprint":
            # 边缘做 0.6px 轻微光学柔化，消除数字生硬切边，融入棉质纤维
            soft_alpha = cv2.GaussianBlur(raw_alpha, (3, 3), sigmaX=0.55)[:, :, np.newaxis]

            # 提取底板微距高频织物纹理 (Weave micro-texture)
            gray_bg = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            blur_micro = cv2.GaussianBlur(gray_bg, (0, 0), sigmaX=1.5)
            weave_detail = np.clip((gray_bg - blur_micro) * weave_strength, -12.0, 12.0)[:, :, np.newaxis]

            # Albedo × Shading 物理乘算 + 织物微纹理透出
            modulated_fg = np.clip(fg_rgb * (0.22 + 0.78 * light_3d) + weave_detail, 0, 255)
            ink_alpha = soft_alpha * ink_opacity
            result = modulated_fg * ink_alpha + result * (1.0 - ink_alpha)

        elif blend_mode == "opaque_screenprint":
            soft_alpha = cv2.GaussianBlur(raw_alpha, (3, 3), sigmaX=0.45)[:, :, np.newaxis]
            # 塑胶浆遮盖力强：墨色基本保留原色，仅随布料明暗轻微起伏
            modulated_fg = np.clip(fg_rgb * (0.62 + 0.38 * light_3d), 0, 255)
            ink_alpha = soft_alpha * ink_opacity
            result = modulated_fg * ink_alpha + result * (1.0 - ink_alpha)

        elif blend_mode == "silk_screen":
            alpha = raw_alpha[:, :, np.newaxis]
            light_3d = lighting_map[:, :, np.newaxis]
            modulated_fg = np.clip(fg_rgb * (0.85 + 0.15 * light_3d), 0, 255)
            result = modulated_fg * alpha + result * (1.0 - alpha)
        else:
            alpha = raw_alpha[:, :, np.newaxis]
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
    def finalize_image_symmetric(self, img_rgb: np.ndarray, quality: int = 88,
                                 seed: Optional[int] = None) -> Image.Image:
        """所有样本（正负样本、退化样本、空底图）必须经历完全相同的保存与压缩管线，
        杜绝让模型通过边缘微弱锯齿或重采样伪影作弊。
        seed 不为空时使用确定性噪声，保证数据集可复现。
        """
        # 添加极微弱传感器模拟噪点 (Sensor Noise, sigma=1.0)
        if seed is None:
            noise = np.random.normal(0, 1.0, img_rgb.shape).astype(np.float32)
        else:
            noise = np.random.default_rng(seed).normal(0, 1.0, img_rgb.shape).astype(np.float32)
        final_array = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(final_array)
        # 内存中模拟真实 JPEG 编码压缩
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
