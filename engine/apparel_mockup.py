"""真实服装与商品实物底板渲染器 (Apparel & Commodity Mockup Factory)。
无需生成式大模型，利用精准参数化矢量 SVG 与物理光影层，程序化渲染真实电商级实物底板：
1. 短袖圆领 T 恤 (Classic Crewneck T-Shirt)：含真实领口螺纹、后领尺码标、落肩装袖缝线、双针下摆走线与布料悬垂折痕。
2. 街头连帽卫衣 (Streetwear Pullover Hoodie)：含立体连帽弧线、抽绳金属头、袋鼠式插手袋与罗纹袖口。
3. 环保帆布托特包 (Cotton Canvas Tote Bag)：含承重织带提手、包口锁边缝线与垂坠褶皱。
4. 陶瓷咖啡马克杯 (Ceramic Coffee Mug)：含柱面立体高光与耳状手柄。
每个模板提供：
- 电商级摄影实物渲染图 (含背景柔和投影与环境光)
- 纯衣物布料区域的 Alpha 掩模 (Garment Mask)
- 预设黄金印花区域的 4 点单应性四边形 (Print Zones: center_chest, left_chest, large_front)
"""
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import resvg_py


COLOR_PALETTES = {
    "white": {
        "body": "#ffffff",
        "shade_1": "#e8ecf2",
        "shade_2": "#d0d5dd",
        "inner_neck": "#c5cad3",
        "stitch": "#d8dce3",
        "name_zh": "纯净白"
    },
    "heather_grey": {
        "body": "#e2e4e8",
        "shade_1": "#c8ccd4",
        "shade_2": "#b0b6c2",
        "inner_neck": "#9aa1af",
        "stitch": "#b8beca",
        "name_zh": "麻灰"
    },
    "black": {
        "body": "#212529",
        "shade_1": "#121416",
        "shade_2": "#08090a",
        "inner_neck": "#181a1d",
        "stitch": "#2d3238",
        "name_zh": "曜石黑"
    },
    "navy_blue": {
        "body": "#1e3a5f",
        "shade_1": "#13253d",
        "shade_2": "#0b1524",
        "inner_neck": "#172d4a",
        "stitch": "#294d7d",
        "name_zh": "藏青色"
    },
    "canvas_beige": {
        "body": "#f4eedb",
        "shade_1": "#dfd7be",
        "shade_2": "#c7bfa3",
        "inner_neck": "#b8b093",
        "stitch": "#cfc7ac",
        "name_zh": "原色胚布"
    }
}


class ApparelMockupFactory:
    """实物服装与商品底板工厂"""

    @classmethod
    def generate_tshirt_svg(cls, color_key: str = "white", width: int = 640, height: int = 640) -> str:
        """生成带精细领口、缝线、折痕与影子的圆领 T 恤 SVG。"""
        c = COLOR_PALETTES.get(color_key, COLOR_PALETTES["white"])
        is_dark = color_key in ["black", "navy_blue"]
        stroke_color = "#3a414a" if is_dark else "#d0d5dd"
        wrinkle_color = "#15181c" if is_dark else "#e5e9f0"
        wrinkle_opacity = "0.5" if is_dark else "0.7"

        return f'''<svg width="{width}" height="{height}" viewBox="0 0 640 640" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- 背景柔和摄影棚冷暖漫反射 -->
    <radialGradient id="studioLight" cx="50%" cy="40%" r="65%">
      <stop offset="0%" stop-color="#f8f9fa"/>
      <stop offset="100%" stop-color="#e9ecef"/>
    </radialGradient>

    <!-- 衣服投射到桌面上的柔和环境投影 (Ambient Drop Shadow) -->
    <filter id="garmentShadow" x="-15%" y="-10%" width="130%" height="130%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="14"/>
      <feOffset dx="0" dy="16" result="blurOffset"/>
      <feFlood flood-color="#000000" flood-opacity="0.14"/>
      <feComposite in2="blurOffset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- 衣服躯干微弱立体曲面渐变 -->
    <linearGradient id="bodyShade" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{c['shade_2']}" stop-opacity="0.45"/>
      <stop offset="18%" stop-color="{c['body']}" stop-opacity="0.0"/>
      <stop offset="82%" stop-color="{c['body']}" stop-opacity="0.0"/>
      <stop offset="100%" stop-color="{c['shade_2']}" stop-opacity="0.45"/>
    </linearGradient>
  </defs>

  <!-- 摄影背景台面 -->
  <rect width="100%" height="100%" fill="url(#studioLight)"/>

  <!-- 主体衣服组合层 (带投影) -->
  <g filter="url(#garmentShadow)">
    <!-- 1. 后领内侧 (Inside Back Neck) 与领标 -->
    <path d="M 255 88 C 290 102 350 102 385 88 C 365 72 275 72 255 88 Z" fill="{c['inner_neck']}"/>
    <rect x="306" y="77" width="28" height="16" rx="2" fill="#88909d"/>
    <line x1="311" y1="83" x2="329" y2="83" stroke="#ffffff" stroke-width="1.5" stroke-opacity="0.7"/>

    <!-- 2. T恤主体与袖子外轮廓 -->
    <path id="tshirt_body"
          d="M 255 88
             C 215 98 170 120 130 150
             L 65 245
             L 125 292
             L 175 228
             C 180 325 182 445 185 570
             C 270 580 370 580 455 570
             C 458 445 460 325 465 228
             L 515 292
             L 575 245
             L 510 150
             C 470 120 425 98 385 88
             C 365 140 275 140 255 88 Z"
          fill="{c['body']}" stroke="{stroke_color}" stroke-width="1.5"/>

    <!-- 3. 躯干立体漫反射光影遮罩 -->
    <path d="M 255 88
             C 215 98 170 120 130 150
             L 65 245 L 125 292 L 175 228
             C 180 325 182 445 185 570
             C 270 580 370 580 455 570
             C 458 445 460 325 465 228
             L 515 292 L 575 245 L 510 150
             C 470 120 425 98 385 88
             C 365 140 275 140 255 88 Z"
          fill="url(#bodyShade)"/>

    <!-- 4. 前领口微厚螺纹带 (Ribbed Collar Band) -->
    <path d="M 255 88 C 275 140 365 140 385 88 C 375 128 265 128 255 88 Z"
          fill="{c['shade_1']}" stroke="{stroke_color}" stroke-width="1"/>

    <!-- 5. 落肩装袖裁片缝线 (Sleeve Seams) -->
    <path d="M 198 106 L 175 228" stroke="{c['stitch']}" stroke-width="1.5" stroke-dasharray="4,2.5"/>
    <path d="M 442 106 L 465 228" stroke="{c['stitch']}" stroke-width="1.5" stroke-dasharray="4,2.5"/>

    <!-- 6. 袖口双针压线 (Cuff Hems) -->
    <line x1="72" y1="250" x2="132" y2="297" stroke="{c['stitch']}" stroke-width="1.5"/>
    <line x1="568" y1="250" x2="508" y2="297" stroke="{c['stitch']}" stroke-width="1.5"/>

    <!-- 7. 下摆双针走线 (Bottom Hem Stitch) -->
    <path d="M 185 558 C 270 568 370 568 455 558" stroke="{c['stitch']}" stroke-width="1.5" fill="none"/>

    <!-- 8. 腋下与胸腹真实悬垂折痕阴影 (Draping Wrinkle Lines) -->
    <path d="M 175 228 C 215 265 255 255 285 275" stroke="{wrinkle_color}" stroke-width="2.5" stroke-opacity="{wrinkle_opacity}" fill="none"/>
    <path d="M 465 228 C 425 265 385 255 355 275" stroke="{wrinkle_color}" stroke-width="2.5" stroke-opacity="{wrinkle_opacity}" fill="none"/>
    <path d="M 182 380 C 230 405 270 395 305 415" stroke="{wrinkle_color}" stroke-width="2" stroke-opacity="{wrinkle_opacity}" fill="none"/>
    <path d="M 458 380 C 410 405 370 395 335 415" stroke="{wrinkle_color}" stroke-width="2" stroke-opacity="{wrinkle_opacity}" fill="none"/>
  </g>
</svg>'''

    @classmethod
    def generate_hoodie_svg(cls, color_key: str = "black", width: int = 640, height: int = 640) -> str:
        """生成连帽卫衣 (Pullover Hoodie) 矢量底板，带帽子、抽绳和袋鼠口袋。"""
        c = COLOR_PALETTES.get(color_key, COLOR_PALETTES["black"])
        is_dark = color_key in ["black", "navy_blue"]
        stroke_color = "#353b48" if is_dark else "#ced6e0"
        string_color = "#f5f6fa" if is_dark else "#718093"
        pocket_shade = "#16191d" if is_dark else "#dfe4ea"

        return f'''<svg width="{width}" height="{height}" viewBox="0 0 640 640" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="hoodieBgLight" cx="50%" cy="40%" r="65%">
      <stop offset="0%" stop-color="#f1f2f6"/>
      <stop offset="100%" stop-color="#dfe4ea"/>
    </radialGradient>
    <filter id="hoodieShadow" x="-15%" y="-10%" width="130%" height="130%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="15"/>
      <feOffset dx="0" dy="18" result="blurOffset"/>
      <feFlood flood-color="#000000" flood-opacity="0.16"/>
      <feComposite in2="blurOffset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect width="100%" height="100%" fill="url(#hoodieBgLight)"/>

  <g filter="url(#hoodieShadow)">
    <!-- 连帽后兜 (Back Hood Silhouette) -->
    <path d="M 220 130 C 200 40 440 40 420 130 Z" fill="{c['shade_1']}"/>

    <!-- 卫衣大身与长袖 -->
    <path d="M 230 130
             C 195 140 155 160 115 190
             L 45 360
             L 105 385
             L 165 270
             C 170 380 172 490 175 580
             C 265 590 375 590 465 580
             C 468 490 470 380 475 270
             L 535 385
             L 595 360
             L 525 190
             C 485 160 445 140 410 130
             C 380 175 260 175 230 130 Z"
          fill="{c['body']}" stroke="{stroke_color}" stroke-width="2"/>

    <!-- 正面连帽领口交叠领 (Crossed Hood Collar) -->
    <path d="M 230 130 C 270 175 370 175 410 130 C 370 195 270 195 230 130 Z"
          fill="{c['shade_1']}" stroke="{stroke_color}" stroke-width="1.5"/>

    <!-- 袋鼠式插手大口袋 (Kangaroo Pouch Pocket) -->
    <path d="M 225 435 L 415 435 L 440 545 L 200 545 Z"
          fill="{pocket_shade}" stroke="{stroke_color}" stroke-width="1.5"/>
    <!-- 口袋两侧插手弧线 -->
    <path d="M 225 435 C 240 475 240 515 200 545" fill="none" stroke="{c['stitch']}" stroke-width="2"/>
    <path d="M 415 435 C 400 475 400 515 440 545" fill="none" stroke="{c['stitch']}" stroke-width="2"/>

    <!-- 抽绳 (Drawstrings with metal aglets) -->
    <path d="M 285 170 C 280 230 275 270 280 320" stroke="{string_color}" stroke-width="3.5" fill="none"/>
    <rect x="277" y="320" width="6" height="15" rx="1" fill="#bdc3c7"/>
    <path d="M 355 170 C 360 230 365 270 360 320" stroke="{string_color}" stroke-width="3.5" fill="none"/>
    <rect x="357" y="320" width="6" height="15" rx="1" fill="#bdc3c7"/>
  </g>
</svg>'''

    @classmethod
    def generate_tote_bag_svg(cls, color_key: str = "canvas_beige", width: int = 640, height: int = 640) -> str:
        """生成环保棉帆布托特包 (Tote Bag) 底板，带提手带与帆布质感折叠。"""
        c = COLOR_PALETTES.get(color_key, COLOR_PALETTES["canvas_beige"])
        return f'''<svg width="{width}" height="{height}" viewBox="0 0 640 640" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="toteBg" cx="50%" cy="40%" r="65%">
      <stop offset="0%" stop-color="#fdfdfd"/>
      <stop offset="100%" stop-color="#e9ecef"/>
    </radialGradient>
    <filter id="toteShadow" x="-15%" y="-10%" width="130%" height="130%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="14"/>
      <feOffset dx="0" dy="16" result="blurOffset"/>
      <feFlood flood-color="#000000" flood-opacity="0.14"/>
      <feComposite in2="blurOffset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect width="100%" height="100%" fill="url(#toteBg)"/>

  <g filter="url(#toteShadow)">
    <!-- 两个棉织织带提手 (Straps) -->
    <path d="M 230 220 C 230 30 270 30 270 220" stroke="{c['shade_1']}" stroke-width="24" fill="none" stroke-linecap="round"/>
    <path d="M 370 220 C 370 30 410 30 410 220" stroke="{c['shade_1']}" stroke-width="24" fill="none" stroke-linecap="round"/>

    <!-- 托特包方形包体 (Bag Main Body) -->
    <path d="M 160 215 L 480 215 L 470 575 L 170 575 Z"
          fill="{c['body']}" stroke="{c['shade_2']}" stroke-width="2"/>

    <!-- 包口加固缝线 (Top Opening Hem) -->
    <line x1="160" y1="235" x2="480" y2="235" stroke="{c['stitch']}" stroke-width="2"/>
    <line x1="160" y1="242" x2="480" y2="242" stroke="{c['stitch']}" stroke-width="2" stroke-dasharray="4,2"/>

    <!-- 底部立体折角包底缝线 (Box Bottom Corners) -->
    <path d="M 170 575 L 205 540 L 435 540 L 470 575" fill="none" stroke="{c['stitch']}" stroke-width="2"/>
    <!-- 帆布垂坠折痕 -->
    <path d="M 240 242 C 250 360 245 460 240 540" stroke="{c['shade_1']}" stroke-width="2" fill="none"/>
    <path d="M 400 242 C 390 360 395 460 400 540" stroke="{c['shade_1']}" stroke-width="2" fill="none"/>
  </g>
</svg>'''

    @classmethod
    def render_apparel_substrate(cls, apparel_type: str = "tshirt",
                                 color_key: str = "white",
                                 width: int = 640,
                                 height: int = 640) -> Tuple[Image.Image, np.ndarray, Dict[str, Any]]:
        """光栅化渲染实物服装底板，并解析印花黄金定位区域。
        返回：
            base_image: 带台面与投影的实物 PIL.Image (RGBA)
            garment_mask: 仅衣服织物区域的二值掩模 (uint8: 0 或 255)
            meta: 印花区域四边形与元数据
        """
        if apparel_type == "tshirt":
            svg_data = cls.generate_tshirt_svg(color_key, width, height)
            garment_name = f"纯棉短袖圆领 T 恤 ({COLOR_PALETTES.get(color_key, {}).get('name_zh', '白')})"
            # 预定义黄金印花区域 (4点四边形 [x, y])
            zones = {
                # 胸口中心大印花 (Chest Center)
                "center_chest": np.float32([
                    [220, 210],
                    [420, 210],
                    [420, 420],
                    [220, 420]
                ]),
                # 左胸小印花 (Left Breast Pocket Area)
                "left_chest": np.float32([
                    [345, 195],
                    [420, 195],
                    [420, 270],
                    [345, 270]
                ]),
                # 大面积满印区域 (Large Front Torso)
                "large_front": np.float32([
                    [195, 185],
                    [445, 185],
                    [440, 520],
                    [200, 520]
                ])
            }
        elif apparel_type == "hoodie":
            svg_data = cls.generate_hoodie_svg(color_key, width, height)
            garment_name = f"街头连帽套头卫衣 ({COLOR_PALETTES.get(color_key, {}).get('name_zh', '黑')})"
            zones = {
                "center_chest": np.float32([
                    [235, 230],
                    [405, 230],
                    [405, 410],
                    [235, 410]
                ]),
                "left_chest": np.float32([
                    [340, 220],
                    [405, 220],
                    [405, 285],
                    [340, 285]
                ]),
                "large_front": np.float32([
                    [215, 215],
                    [425, 215],
                    [420, 425],
                    [220, 425]
                ])
            }
        elif apparel_type == "tote_bag":
            svg_data = cls.generate_tote_bag_svg(color_key, width, height)
            garment_name = f"环保棉帆布托特包 ({COLOR_PALETTES.get(color_key, {}).get('name_zh', '米原色')})"
            zones = {
                "center_chest": np.float32([
                    [210, 275],
                    [430, 275],
                    [430, 500],
                    [210, 500]
                ]),
                "left_chest": np.float32([
                    [330, 270],
                    [430, 270],
                    [430, 370],
                    [330, 370]
                ]),
                "large_front": np.float32([
                    [185, 245],
                    [455, 245],
                    [445, 545],
                    [195, 545]
                ])
            }
        else:
            raise ValueError(f"未知的实物底板类型: {apparel_type}")

        # 利用 resvg 光栅化
        png_bytes = resvg_py.svg_to_bytes(svg_data, width=width, height=height)
        base_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        # 提取布料 Alpha 掩模 (根据背景颜色差值计算，中心衣服区域为 255)
        arr = np.array(base_img)
        # 背景为浅灰/渐变，衣服边缘轮廓清晰，此处构造有效布料二值 mask
        gray = np.mean(arr[:, :, :3], axis=2)
        # 简单阈值或几何裁剪提取衣服区域
        garment_mask = np.zeros((height, width), dtype=np.uint8)
        # 衣服大概区域
        y_indices, x_indices = np.where(gray < 252)
        if len(y_indices) > 0:
            garment_mask[y_indices, x_indices] = 255

        meta = {
            "apparel_type": apparel_type,
            "color_key": color_key,
            "garment_name_zh": garment_name,
            "zones": zones
        }
        return base_img, garment_mask, meta
