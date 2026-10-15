"""参数化几何图元绘制引擎。
采用纯数学极坐标与几何多边形计算，生成像素级精确的矢量几何图形。
"""
import math
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def draw_star(draw: ImageDraw.ImageDraw, center: Tuple[int, int], outer_r: float,
              inner_r: float, points: int = 5, rotation: float = -math.pi / 2,
              fill: Tuple[int, int, int] = (255, 204, 0), outline=None) -> List[Tuple[float, float]]:
    """绘制正多角星（如五角星、四角星、八角星），支持任意旋转角度。"""
    cx, cy = center
    angle_step = math.pi / points
    vertices = []
    for i in range(2 * points):
        r = outer_r if i % 2 == 0 else inner_r
        angle = rotation + i * angle_step
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        vertices.append((x, y))
    draw.polygon(vertices, fill=fill, outline=outline)
    return vertices


def draw_crescent(image: Image.Image, center: Tuple[int, int], radius: float,
                  offset_ratio: float = 0.35, inner_scale: float = 0.85,
                  angle_deg: float = 0, fill: Tuple[int, int, int] = (255, 255, 255)):
    """绘制平滑月牙（新月/弯月）：使用透明遮罩布尔差集精确切削。"""
    cx, cy = center
    mask = Image.new("L", image.size, 0)
    draw_mask = ImageDraw.Draw(mask)

    # 1. 外圆
    draw_mask.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)

    # 2. 内切圆（根据角度偏移切削）
    rad = math.radians(angle_deg)
    offset_dist = radius * offset_ratio
    in_cx = cx + offset_dist * math.cos(rad)
    in_cy = cy + offset_dist * math.sin(rad)
    in_r = radius * inner_scale
    draw_mask.ellipse([in_cx - in_r, in_cy - in_r, in_cx + in_r, in_cy + in_r], fill=0)

    # 3. 复合到目标图层
    crescent_layer = Image.new("RGBA", image.size, fill + (255,))
    image.paste(crescent_layer, (0, 0), mask=mask)


def draw_sun_with_rays(draw: ImageDraw.ImageDraw, center: Tuple[int, int],
                       core_radius: float, ray_length: float, num_rays: int = 16,
                       fill: Tuple[int, int, int] = (255, 204, 0)):
    """绘制太阳与放射状尖芒。"""
    cx, cy = center
    # 绘制光芒尖刺
    angle_step = 2 * math.pi / num_rays
    for i in range(num_rays):
        base_angle = i * angle_step
        tip_x = cx + (core_radius + ray_length) * math.cos(base_angle)
        tip_y = cy + (core_radius + ray_length) * math.sin(base_angle)

        w_angle = angle_step * 0.3
        p1_x = cx + core_radius * math.cos(base_angle - w_angle)
        p1_y = cy + core_radius * math.sin(base_angle - w_angle)
        p2_x = cx + core_radius * math.cos(base_angle + w_angle)
        p2_y = cy + core_radius * math.sin(base_angle + w_angle)

        draw.polygon([(tip_x, tip_y), (p1_x, p1_y), (p2_x, p2_y)], fill=fill)

    # 绘制中心太阳圆盘
    draw.ellipse([cx - core_radius, cy - core_radius, cx + core_radius, cy + core_radius], fill=fill)


def draw_shield(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int],
                fill: Tuple[int, int, int], outline: Tuple[int, int, int] = (0, 0, 0),
                outline_width: int = 4):
    """绘制纹章盾徽基底轮廓。"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # 盾牌顶点与弧线多边形逼近
    points = [
        (x1, y1),
        (x2, y1),
        (x2, y1 + h * 0.6),
        (x1 + w * 0.5, y2),
        (x1, y1 + h * 0.6)
    ]
    draw.polygon(points, fill=fill, outline=outline, width=outline_width)


def draw_ribbon_with_text(image: Image.Image, center: Tuple[int, int],
                          text: str, fill: Tuple[int, int, int] = (255, 255, 255),
                          text_color: Tuple[int, int, int] = (0, 0, 0)):
    """绘制徽章飘带及铭文文字。"""
    draw = ImageDraw.Draw(image)
    cx, cy = center
    # 动态估计文字尺寸
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    pad_x, pad_y = 16, 6
    rw, rh = tw + pad_x * 2, th + pad_y * 2
    rx1, ry1 = cx - rw // 2, cy - rh // 2
    rx2, ry2 = cx + rw // 2, cy + rh // 2

    # 飘带主体与折角轮廓
    draw.rectangle([rx1, ry1, rx2, ry2], fill=fill, outline=(80, 80, 80), width=2)
    # 绘制飘带两端燕尾缺角
    draw.polygon([(rx1, ry1), (rx1 - 12, cy), (rx1, ry2)], fill=fill, outline=(80, 80, 80))
    draw.polygon([(rx2, ry1), (rx2 + 12, cy), (rx2, ry2)], fill=fill, outline=(80, 80, 80))

    # 居中写入文字
    draw.text((cx - tw // 2, cy - th // 2 - 2), text, font=font, fill=text_color)
    return [int(rx1 - 12), int(ry1), int(rx2 + 12), int(ry2)]
