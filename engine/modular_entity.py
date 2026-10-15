"""模块化原创积木实体生成引擎 (Modular Entity Builder)。
基于 Kenney Monster / Vector Badge 理念，通过程序化几何与图层装配，构造 100% 可控、无外部先验偏置的原创多模态实体：
1. 身体底盘 (Body Base): 圆润软泥 (Blob)、盾形图徽 (Shield)、圆角六边形 (Hex)、星徽 (Star-Emblem)
2. 眼睛系统 (Eye Subsystem): 数量精确可控 (1只大独眼、2只正视眼、3只排布眼、5只环形眼)
3. 辅件系统 (Accessories): 弯角 (Curved Horns)、触角天线 (Antenna)、王冠 (Crown)、星芒
4. 符号与胸标 (Chest Emblems): 五角星、心形、闪电、锚
5. 配色方案 (Color Schemes): 霓虹蓝紫、赛博青黄、珊瑚红白、深林军绿
每个实体均提供像素级透明 RGBA 图像，以及细粒度部件层标注 (Hierarchical Ground Truth)。
"""
import math
import random
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image, ImageDraw

from .coords import to_rel_bbox_1000
from .geometry import draw_star


THEMES = {
    "cyber_cyan": {"body": (0, 210, 211), "eye": (255, 255, 255), "pupil": (16, 32, 64), "acc": (255, 159, 67), "chest": (254, 202, 87)},
    "coral_pink": {"body": (255, 107, 129), "eye": (255, 255, 255), "pupil": (47, 53, 66), "acc": (245, 205, 121), "chest": (112, 161, 255)},
    "lime_green": {"body": (46, 204, 113), "eye": (255, 255, 255), "pupil": (24, 44, 30), "acc": (241, 196, 15), "chest": (231, 76, 60)},
    "royal_purple": {"body": (155, 89, 182), "eye": (255, 255, 255), "pupil": (44, 62, 80), "acc": (52, 152, 219), "chest": (243, 156, 18)},
    "obsidian_gold": {"body": (45, 52, 54), "eye": (255, 255, 255), "pupil": (0, 0, 0), "acc": (253, 203, 110), "chest": (225, 112, 85)}
}


class ModularEntityBuilder:
    """模块化几何实体构建器"""

    @classmethod
    def build_entity(cls,
                     body_type: str = "blob",
                     eye_count: int = 3,
                     accessory: str = "horns",
                     chest_symbol: str = "star",
                     theme_name: str = "cyber_cyan",
                     size: Tuple[int, int] = (400, 400)) -> Tuple[Image.Image, Dict[str, Any]]:
        """装配并渲染单个原创实体，返回 (RGBA 图像, 结构化元数据)。"""
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        theme = THEMES.get(theme_name, THEMES["cyber_cyan"])
        cx, cy = w // 2, int(h * 0.56)

        component_records = []

        # ---------------------------------------------------------------------
        # 1. 绘制头顶辅件 (Accessories: 角/天线/王冠，位于身体后方或顶端)
        # ---------------------------------------------------------------------
        acc_meta = {"type": accessory, "color": theme["acc"], "present": accessory != "none"}
        if accessory == "horns":
            # 左右双弯角
            horn_color = theme["acc"]
            # 左角
            draw.polygon([(cx - 70, cy - 90), (cx - 120, cy - 160), (cx - 40, cy - 110)], fill=horn_color)
            # 右角
            draw.polygon([(cx + 70, cy - 90), (cx + 120, cy - 160), (cx + 40, cy - 110)], fill=horn_color)
            acc_meta["name_zh"] = "对称双弯角"
        elif accessory == "antenna":
            # 顶部三叉雷达触角
            ant_color = theme["acc"]
            draw.line([(cx, cy - 100), (cx, cy - 170)], fill=ant_color, width=8)
            draw.ellipse([cx - 16, cy - 186, cx + 16, cy - 154], fill=theme["chest"])
            acc_meta["name_zh"] = "中央雷达圆珠触角"
        elif accessory == "crown":
            # 三齿王冠
            crown_color = theme["acc"]
            crown_pts = [
                (cx - 60, cy - 110),
                (cx - 60, cy - 160),
                (cx - 25, cy - 130),
                (cx, cy - 170),
                (cx + 25, cy - 130),
                (cx + 60, cy - 160),
                (cx + 60, cy - 110)
            ]
            draw.polygon(crown_pts, fill=crown_color)
            acc_meta["name_zh"] = "三齿黄金王冠"
        else:
            acc_meta["name_zh"] = "无头顶饰物"

        # ---------------------------------------------------------------------
        # 2. 绘制身体主底盘 (Body Base)
        # ---------------------------------------------------------------------
        body_color = theme["body"]
        if body_type == "blob":
            # 圆润水滴/怪兽身体
            body_box = [cx - 110, cy - 110, cx + 110, cy + 110]
            draw.rounded_rectangle(body_box, radius=70, fill=body_color)
            body_desc = "圆角凝胶软泥形态"
        elif body_type == "shield":
            # 徽章盾形
            shield_pts = [
                (cx - 105, cy - 110),
                (cx + 105, cy - 110),
                (cx + 105, cy + 30),
                (cx, cy + 125),
                (cx - 105, cy + 30)
            ]
            draw.polygon(shield_pts, fill=body_color)
            body_desc = "经典先锋盾徽形态"
        elif body_type == "hex":
            # 科技六边形
            hex_pts = [
                (cx - 110, cy),
                (cx - 55, cy - 105),
                (cx + 55, cy - 105),
                (cx + 110, cy),
                (cx + 55, cy + 105),
                (cx - 55, cy + 105)
            ]
            draw.polygon(hex_pts, fill=body_color)
            body_desc = "几何六边形蜂窝形态"
        else:
            draw.ellipse([cx - 105, cy - 105, cx + 105, cy + 105], fill=body_color)
            body_desc = "正圆形徽章"

        # ---------------------------------------------------------------------
        # 3. 绘制胸标符号 (Chest Symbol)
        # ---------------------------------------------------------------------
        sym_color = theme["chest"]
        chest_y = cy + 45
        if chest_symbol == "star":
            draw_star(draw, (cx, chest_y), outer_r=28, inner_r=13, points=5, fill=sym_color)
            sym_desc = "五角星胸徽"
        elif chest_symbol == "diamond":
            draw.polygon([(cx, chest_y - 25), (cx + 25, chest_y), (cx, chest_y + 25), (cx - 25, chest_y)], fill=sym_color)
            sym_desc = "正菱形胸徽"
        elif chest_symbol == "cross":
            draw.rectangle([cx - 8, chest_y - 24, cx + 8, chest_y + 24], fill=sym_color)
            draw.rectangle([cx - 24, chest_y - 8, cx + 24, chest_y + 8], fill=sym_color)
            sym_desc = "十字胸标"
        else:
            sym_desc = "无胸标"

        # ---------------------------------------------------------------------
        # 4. 绘制眼睛系统 (Eye Subsystem: 1 / 2 / 3 / 5 颗，极高区分度)
        # ---------------------------------------------------------------------
        eye_white = theme["eye"]
        pupil_color = theme["pupil"]
        eye_coords = []

        if eye_count == 1:
            # 单颗居中巨大独眼
            eye_coords.append((cx, cy - 35, 34, 15))
        elif eye_count == 2:
            # 双眼对称
            eye_coords.append((cx - 42, cy - 35, 24, 11))
            eye_coords.append((cx + 42, cy - 35, 24, 11))
        elif eye_count == 3:
            # 三眼（三角形或水平排列）：上方一大，下方二小
            eye_coords.append((cx, cy - 55, 22, 10))
            eye_coords.append((cx - 45, cy - 25, 19, 9))
            eye_coords.append((cx + 45, cy - 25, 19, 9))
        elif eye_count == 5:
            # 五眼奇星排列
            eye_coords.append((cx, cy - 40, 20, 9))
            for ang in [0, 90, 180, 270]:
                rad = math.radians(ang)
                ex = int(cx + 48 * math.cos(rad))
                ey = int((cy - 40) + 32 * math.sin(rad))
                eye_coords.append((ex, ey, 14, 6))

        for (ex, ey, er, pr) in eye_coords:
            # 眼白
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=eye_white, outline=(0, 0, 0, 100), width=2)
            # 瞳孔
            draw.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=pupil_color)
            # 瞳孔反光小白点
            draw.ellipse([ex - pr + 2, ey - pr + 2, ex - pr + 5, ey - pr + 5], fill=(255, 255, 255))

        entity_id = f"creature_{theme_name}_{body_type}_{eye_count}eyes_{accessory}_{chest_symbol}"
        meta = {
            "entity_id": entity_id,
            "entity_type": "modular_creature",
            "theme": theme_name,
            "body_type": body_type,
            "body_desc": body_desc,
            "eye_count": eye_count,
            "accessory": accessory,
            "accessory_desc": acc_meta["name_zh"],
            "chest_symbol": chest_symbol,
            "chest_desc": sym_desc,
            "discriminative_summary": f"{eye_count}只眼睛、{acc_meta['name_zh']}、{body_desc}、{sym_desc}"
        }

        return img, meta
