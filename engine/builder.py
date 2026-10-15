"""底板拓扑与元素编排器。
支持多种国际旗帜标准版式（横条纹、竖条纹、Canton留白、北欧十字、对角分割）
以及中心/偏位符号、星星排布、太阳月牙、徽章的参数化组装。
"""
import math
import random
from typing import Dict, Any, List
from PIL import Image, ImageDraw

from .palette import COLORS, get_color
from .coords import to_rel_bbox_1000
from .geometry import (
    draw_star, draw_crescent, draw_sun_with_rays,
    draw_shield, draw_ribbon_with_text
)


class FlagBuilder:
    def __init__(self, width: int = 600, height: int = 400):
        self.width = width
        self.height = height

    def create_base(self, layout_type: str, color_keys: List[str]) -> Tuple[Image.Image, Dict[str, Any]]:
        img = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        meta = {
            "layout": layout_type,
            "base_colors": [get_color(k)["name"] for k in color_keys],
            "base_color_keys": color_keys
        }

        if layout_type == "horizontal_triband":
            # 三等分横条纹（如德、法式横、荷）
            h_step = self.height / 3.0
            for i, c_key in enumerate(color_keys[:3]):
                draw.rectangle([0, int(i * h_step), self.width, int((i + 1) * h_step)],
                               fill=get_color(c_key)["rgb"])
            meta["layout_desc"] = "三等分水平横条纹，从上到下排列"

        elif layout_type == "vertical_triband":
            # 三等分竖条纹（如法、意）
            w_step = self.width / 3.0
            for i, c_key in enumerate(color_keys[:3]):
                draw.rectangle([int(i * w_step), 0, int((i + 1) * w_step), self.height],
                               fill=get_color(c_key)["rgb"])
            meta["layout_desc"] = "三等分垂直竖条纹，从左到右排列"

        elif layout_type == "horizontal_biband":
            # 二等分横条纹（如波兰、印尼）
            h_step = self.height / 2.0
            for i, c_key in enumerate(color_keys[:2]):
                draw.rectangle([0, int(i * h_step), self.width, int((i + 1) * h_step)],
                               fill=get_color(c_key)["rgb"])
            meta["layout_desc"] = "二等分水平横条纹，上下均分"

        elif layout_type == "canton":
            # 条纹底 + 左上角 Canton 方块（如美、希腊）
            num_stripes = 7
            stripe_h = self.height / num_stripes
            c1 = get_color(color_keys[0])["rgb"]
            c2 = get_color(color_keys[1])["rgb"]
            for i in range(num_stripes):
                c = c1 if i % 2 == 0 else c2
                draw.rectangle([0, int(i * stripe_h), self.width, int((i + 1) * stripe_h)], fill=c)
            # 绘制 Canton 方块 (占宽 40%，高 4/7)
            canton_w = int(self.width * 0.4)
            canton_h = int(stripe_h * 4)
            canton_color = get_color(color_keys[2] if len(color_keys) > 2 else "blue")
            draw.rectangle([0, 0, canton_w, canton_h], fill=canton_color["rgb"])
            meta["canton_box"] = to_rel_bbox_1000([0, 0, canton_h, canton_w], self.width, self.height)
            meta["layout_desc"] = f"多道水平条纹底板，左上角带有{canton_color['name']}的方形区域(Canton)"

        elif layout_type == "nordic_cross":
            # 北欧十字（偏左十字，如瑞典、挪威）
            bg_color = get_color(color_keys[0])["rgb"]
            cross_color = get_color(color_keys[1])["rgb"]
            draw.rectangle([0, 0, self.width, self.height], fill=bg_color)
            cross_w = int(self.height * 0.22)
            cross_x = int(self.width * 0.35)
            # 纵向十字条
            draw.rectangle([cross_x - cross_w // 2, 0, cross_x + cross_w // 2, self.height], fill=cross_color)
            # 横向十字条
            cy = self.height // 2
            draw.rectangle([0, cy - cross_w // 2, self.width, cy + cross_w // 2], fill=cross_color)
            meta["layout_desc"] = "北欧十字版式（纵轴靠左侧旗杆端偏置）"

        elif layout_type == "diagonal_bend":
            # 对角斜切（如坦桑尼亚）
            c_bg1 = get_color(color_keys[0])["rgb"]
            c_bg2 = get_color(color_keys[1])["rgb"]
            c_stripe = get_color(color_keys[2] if len(color_keys) > 2 else "black")["rgb"]
            draw.polygon([(0, 0), (self.width, 0), (self.width, self.height)], fill=c_bg1)
            draw.polygon([(0, 0), (0, self.height), (self.width, self.height)], fill=c_bg2)
            # 中间粗斜条纹
            band_w = int(self.height * 0.3)
            draw.polygon([(0, self.height), (band_w, self.height), (self.width, band_w),
                          (self.width, 0), (self.width - band_w, 0), (0, self.height - band_w)], fill=c_stripe)
            meta["layout_desc"] = "对角斜切分割版式，包含贯穿对角的粗斜带"

        else:
            # 纯色底（solid_field）
            bg_c = get_color(color_keys[0])["rgb"]
            draw.rectangle([0, 0, self.width, self.height], fill=bg_c)
            meta["layout_desc"] = f"单色纯色旗帜基底，颜色为{get_color(color_keys[0])['name']}"

        return img, meta

    def add_emblem_or_symbols(self, img: Image.Image, element_type: str,
                              element_color: str, position: str = "center",
                              count: int = 1, text: str = None) -> Dict[str, Any]:
        """向旗帜添加符号/徽章，并记录绝对坐标、方位、数量与属性。"""
        draw = ImageDraw.Draw(img)
        w, h = self.width, self.height
        c_rgb = get_color(element_color)["rgb"]
        c_name = get_color(element_color)["name"]

        # 解析摆放中心与范围
        if position == "canton":
            cx, cy = int(w * 0.2), int(h * 0.28)
            pos_desc = "左上角 (Canton区域)"
        elif position == "hoist":
            cx, cy = int(w * 0.25), int(h * 0.5)
            pos_desc = "靠旗杆侧（左侧偏中）"
        elif position == "top_left":
            cx, cy = int(w * 0.15), int(h * 0.2)
            pos_desc = "左上方"
        else:
            cx, cy = int(w * 0.5), int(h * 0.5)
            pos_desc = "正中央"

        elem_meta = {
            "element_type": element_type,
            "element_color": c_name,
            "position": pos_desc,
            "count": count,
            "elements_detail": []
        }

        if element_type == "stars":
            # 星星识别与数量识别
            if count == 1:
                r_out = int(h * 0.18) if position == "center" else int(h * 0.12)
                draw_star(draw, (cx, cy), outer_r=r_out, inner_r=r_out * 0.38, fill=c_rgb)
                elem_meta["elements_detail"].append({
                    "name": "五角星",
                    "bbox": to_rel_bbox_1000([cy - r_out, cx - r_out, cy + r_out, cx + r_out], w, h)
                })
            elif count == 3:
                # 3 颗水平排开
                spacing = int(w * 0.12)
                r_out = int(h * 0.08)
                for idx, offset in enumerate([-1, 0, 1]):
                    scx = cx + offset * spacing
                    draw_star(draw, (scx, cy), outer_r=r_out, inner_r=r_out * 0.38, fill=c_rgb)
                    elem_meta["elements_detail"].append({
                        "name": f"第{idx+1}颗五角星",
                        "bbox": to_rel_bbox_1000([cy - r_out, scx - r_out, cy + r_out, scx + r_out], w, h)
                    })
            elif count == 5:
                # 拱形/环形排列五星
                orbit_r = int(h * 0.18)
                star_r = int(h * 0.06)
                for idx in range(5):
                    angle = -math.pi * 0.8 + idx * (math.pi * 0.4)
                    sx = int(cx + orbit_r * math.cos(angle))
                    sy = int(cy + orbit_r * math.sin(angle))
                    draw_star(draw, (sx, sy), outer_r=star_r, inner_r=star_r * 0.38,
                              rotation=angle + math.pi/2, fill=c_rgb)
                    elem_meta["elements_detail"].append({
                        "name": f"环形第{idx+1}颗星",
                        "bbox": to_rel_bbox_1000([sy - star_r, sx - star_r, sy + star_r, sx + star_r], w, h)
                    })

        elif element_type == "crescent_and_star":
            # 新月与星组合（如伊斯兰国家旗帜元素）
            cr_radius = int(h * 0.18)
            draw_crescent(img, (cx, cy), radius=cr_radius, fill=c_rgb, angle_deg=25)
            # 在新月开口处放置一颗小星
            star_cx = int(cx + cr_radius * 0.55)
            star_cy = cy
            star_r = int(cr_radius * 0.38)
            draw_star(draw, (star_cx, star_cy), outer_r=star_r, inner_r=star_r * 0.38,
                      rotation=math.radians(20), fill=c_rgb)
            elem_meta["elements_detail"].append({
                "name": "弯月",
                "bbox": to_rel_bbox_1000([cy - cr_radius, cx - cr_radius, cy + cr_radius, cx + cr_radius], w, h)
            })
            elem_meta["elements_detail"].append({
                "name": "伴星",
                "bbox": to_rel_bbox_1000([star_cy - star_r, star_cx - star_r, star_cy + star_r, star_cx + star_r], w, h)
            })

        elif element_type == "sun":
            # 太阳与光芒
            core_r = int(h * 0.11)
            ray_len = int(h * 0.08)
            num_rays = 16
            draw_sun_with_rays(draw, (cx, cy), core_radius=core_r, ray_length=ray_len,
                               num_rays=num_rays, fill=c_rgb)
            elem_meta["rays_count"] = num_rays
            total_r = core_r + ray_len
            elem_meta["elements_detail"].append({
                "name": f"{num_rays}道光芒太阳",
                "bbox": to_rel_bbox_1000([cy - total_r, cx - total_r, cy + total_r, cx + total_r], w, h)
            })

        elif element_type == "coat_of_arms":
            # 复合国徽徽章：盾牌 + 内部图腾 + 下方飘带文字
            shield_w, shield_h = int(w * 0.22), int(h * 0.38)
            s_bbox = (cx - shield_w // 2, cy - shield_h // 2 - 10,
                      cx + shield_w // 2, cy + shield_h // 2 - 10)
            draw_shield(draw, s_bbox, fill=c_rgb, outline=(255, 255, 255), outline_width=3)
            # 盾牌内嵌小十字或小星
            inner_r = int(shield_h * 0.16)
            draw_star(draw, (cx, cy - 12), outer_r=inner_r, inner_r=inner_r * 0.4,
                      fill=(255, 255, 255))
            shield_rel = to_rel_bbox_1000([s_bbox[1], s_bbox[0], s_bbox[3], s_bbox[2]], w, h)
            elem_meta["elements_detail"].append({"name": "中央纹章盾牌", "bbox": shield_rel})

            # 飘带文字（OCR）
            if text:
                ribbon_bbox = draw_ribbon_with_text(img, (cx, cy + shield_h // 2 + 10), text=text)
                ribbon_rel = to_rel_bbox_1000([ribbon_bbox[1], ribbon_bbox[0], ribbon_bbox[3], ribbon_bbox[2]], w, h)
                elem_meta["text"] = text
                elem_meta["text_bbox"] = ribbon_rel
                elem_meta["elements_detail"].append({"name": f"飘带文字'{text}'", "bbox": ribbon_rel})

        return elem_meta
