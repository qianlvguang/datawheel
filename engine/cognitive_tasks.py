"""高阶认知推理生成引擎。
包含：
1. 数星星与密集星群计数 (Star Counting & Grouping)
2. 对比图与找不同 (Spot the Difference / Comparative Dual-Panel)
3. 视觉符号加减法 (Visual Arithmetic: A+B, |A-B|)
4. 多增少补 (Surplus & Deficit / Target Balancing: 规则比对与多退少补)
全部输出的 BBox 统一采用以左上角为原点、0~1000 的相对坐标系 [ymin, xmin, ymax, xmax]。
"""
import random
import math
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageDraw

from .palette import COLORS, get_color
from .geometry import draw_star
from .coords import to_rel_bbox_1000


class CognitiveTaskEngine:

    @staticmethod
    def generate_star_counting_task(width: int = 600, height: int = 400,
                                   star_count: int = None,
                                   bg_color_key: str = "navy_blue",
                                   star_color_key: str = "gold") -> Tuple[Image.Image, Dict[str, Any]]:
        """1. 数星星与星群分布生成器"""
        if star_count is None:
            star_count = random.randint(4, 12)

        img = Image.new("RGBA", (width, height), get_color(bg_color_key)["rgb"])
        draw = ImageDraw.Draw(img)
        star_rgb = get_color(star_color_key)["rgb"]
        star_cname = get_color(star_color_key)["name"]

        # 随机分布在画面内，保证不越界且不严重重叠
        stars_info = []
        occupied_centers = []
        radius = int(height * 0.08)

        # 尝试生成不重叠的星星位置
        for i in range(star_count):
            placed = False
            for _ in range(50):
                cx = random.randint(int(width * 0.12), int(width * 0.88))
                cy = random.randint(int(height * 0.15), int(height * 0.85))
                # 检查距离
                if all(math.hypot(cx - ox, cy - oy) > radius * 2.2 for ox, oy in occupied_centers):
                    occupied_centers.append((cx, cy))
                    placed = True
                    break
            if not placed:
                cx = random.randint(int(width * 0.1), int(width * 0.9))
                cy = random.randint(int(height * 0.1), int(height * 0.9))
                occupied_centers.append((cx, cy))

            # 随机轻微旋转
            rot = random.uniform(-math.pi, math.pi)
            draw_star(draw, (cx, cy), outer_r=radius, inner_r=radius * 0.38,
                      rotation=rot, fill=star_rgb)

            # 0~1000 相对坐标
            rel_box = to_rel_bbox_1000([cy - radius, cx - radius, cy + radius, cx + radius], width, height)
            stars_info.append({
                "star_id": i + 1,
                "center_pixel": [cx, cy],
                "bbox_1000": rel_box,
                "color": star_cname
            })

        meta = {
            "task_type": "star_counting",
            "bg_color": get_color(bg_color_key)["name"],
            "star_color": star_cname,
            "star_count": star_count,
            "stars_info": stars_info,
            "all_bboxes_1000": [s["bbox_1000"] for s in stars_info]
        }
        return img, meta

    @staticmethod
    def generate_spot_the_difference_pair(width: int = 400, height: int = 400,
                                          diff_type: str = "missing_element") -> Tuple[Image.Image, Dict[str, Any]]:
        """2. 对比图与找不同（左右双联画）：精准定位差异区域并在 0~1000 标出"""
        bg_rgb = get_color("navy_blue")["rgb"]
        star_gold = get_color("gold")["rgb"]
        star_red = get_color("crimson")["rgb"]

        # 左图：基准图 (5 颗星，环形排列)
        img_left = Image.new("RGBA", (width, height), bg_rgb)
        draw_l = ImageDraw.Draw(img_left)
        orbit_r = int(height * 0.28)
        star_r = int(height * 0.09)
        cx, cy = width // 2, height // 2

        coords = []
        for i in range(5):
            angle = -math.pi / 2 + i * (2 * math.pi / 5)
            sx = int(cx + orbit_r * math.cos(angle))
            sy = int(cy + orbit_r * math.sin(angle))
            draw_star(draw_l, (sx, sy), outer_r=star_r, inner_r=star_r * 0.38,
                      rotation=angle + math.pi/2, fill=star_gold)
            coords.append((sx, sy))

        # 右图：发生突变的图
        img_right = Image.new("RGBA", (width, height), bg_rgb)
        draw_r = ImageDraw.Draw(img_right)

        diff_idx = 2  # 改变第 3 颗星 (右下方那颗)
        diff_sx, diff_sy = coords[diff_idx]
        diff_desc = ""

        for i, (sx, sy) in enumerate(coords):
            if i == diff_idx:
                if diff_type == "missing_element":
                    # 缺少一颗星
                    diff_desc = "右图右下角缺少了一颗原本存在的金色五角星（该位置变为空白）"
                    continue
                elif diff_type == "color_changed":
                    # 变色
                    draw_star(draw_r, (sx, sy), outer_r=star_r, inner_r=star_r * 0.38,
                              rotation=-math.pi / 2 + i * (2 * math.pi / 5) + math.pi/2, fill=star_red)
                    diff_desc = "右图右下角的五角星由原本的金色变成了绯红色"
            else:
                draw_star(draw_r, (sx, sy), outer_r=star_r, inner_r=star_r * 0.38,
                          rotation=-math.pi / 2 + i * (2 * math.pi / 5) + math.pi/2, fill=star_gold)

        # 左右水平拼接为一张总对比图 (双联画)
        total_w = width * 2 + 10
        total_h = height
        composite = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        composite.paste(img_left, (0, 0))
        composite.paste(img_right, (width + 10, 0))

        # 分割线
        draw_comp = ImageDraw.Draw(composite)
        draw_comp.rectangle([width, 0, width + 10, total_h], fill=(180, 180, 180, 255))

        # 计算差异区域在总对比图 (0~1000 相对坐标系) 中的位置
        # 左侧对应区域
        diff_box_left_px = [diff_sy - star_r, diff_sx - star_r, diff_sy + star_r, diff_sx + star_r]
        diff_box_left_1000 = to_rel_bbox_1000(diff_box_left_px, total_w, total_h)

        # 右侧对应区域
        diff_box_right_px = [diff_sy - star_r, width + 10 + diff_sx - star_r,
                             diff_sy + star_r, width + 10 + diff_sx + star_r]
        diff_box_right_1000 = to_rel_bbox_1000(diff_box_right_px, total_w, total_h)

        meta = {
            "task_type": "spot_the_difference",
            "diff_type": diff_type,
            "difference_description": diff_desc,
            "target_differing_box_left_1000": diff_box_left_1000,
            "target_differing_box_right_1000": diff_box_right_1000,
            "left_star_count": 5,
            "right_star_count": 4 if diff_type == "missing_element" else 5
        }
        return composite, meta

    @staticmethod
    def generate_visual_arithmetic_pair(width: int = 400, height: int = 400,
                                        count_a: int = 3, count_b: int = 5) -> Tuple[Image.Image, Dict[str, Any]]:
        """3. 视觉加减法：左图 A 颗星，右图 B 颗星，计算 A+B 和 |A-B|"""
        c_bg = get_color("dark_red")["rgb"]
        c_star = get_color("yellow")["rgb"]

        # 左图
        img_a = Image.new("RGBA", (width, height), c_bg)
        draw_a = ImageDraw.Draw(img_a)
        r = int(height * 0.1)
        step_a = width / (count_a + 1)
        boxes_a = []
        for i in range(count_a):
            cx, cy = int((i + 1) * step_a), height // 2
            draw_star(draw_a, (cx, cy), outer_r=r, inner_r=r * 0.38, fill=c_star)
            boxes_a.append([cy - r, cx - r, cy + r, cx + r])

        # 右图
        img_b = Image.new("RGBA", (width, height), c_bg)
        draw_b = ImageDraw.Draw(img_b)
        step_b = width / (count_b + 1)
        boxes_b = []
        for i in range(count_b):
            cx, cy = int((i + 1) * step_b), height // 2
            draw_star(draw_b, (cx, cy), outer_r=r, inner_r=r * 0.38, fill=c_star)
            boxes_b.append([cy - r, cx - r, cy + r, cx + r])

        # 拼接
        total_w = width * 2 + 10
        total_h = height
        composite = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        composite.paste(img_a, (0, 0))
        composite.paste(img_b, (width + 10, 0))
        draw_comp = ImageDraw.Draw(composite)
        draw_comp.rectangle([width, 0, width + 10, total_h], fill=(180, 180, 180, 255))

        # 转换为全图 0~1000 坐标
        rel_boxes_a = [to_rel_bbox_1000(b, total_w, total_h) for b in boxes_a]
        rel_boxes_b = [to_rel_bbox_1000([b[0], b[1] + width + 10, b[2], b[3] + width + 10], total_w, total_h) for b in boxes_b]

        meta = {
            "task_type": "visual_arithmetic",
            "left_count": count_a,
            "right_count": count_b,
            "sum": count_a + count_b,
            "diff": abs(count_b - count_a),
            "left_bboxes_1000": rel_boxes_a,
            "right_bboxes_1000": rel_boxes_b,
            "arithmetic_expr": f"左图({count_a}颗) + 右图({count_b}颗) = 合计{count_a + count_b}颗；差值 = {abs(count_b - count_a)}颗"
        }
        return composite, meta

    @staticmethod
    def generate_surplus_deficit_task(width: int = 600, height: int = 400,
                                      target_standard_count: int = 5,
                                      actual_count: int = 3) -> Tuple[Image.Image, Dict[str, Any]]:
        """4. 多增少补 (Target Balancing)：对照目标标准数量，判断是多增还是缺少"""
        img = Image.new("RGBA", (width, height), get_color("dark_red")["rgb"])
        draw = ImageDraw.Draw(img)
        c_star = get_color("yellow")["rgb"]

        r = int(height * 0.12)
        step = width / (actual_count + 1)
        stars_meta = []
        for i in range(actual_count):
            cx, cy = int((i + 1) * step), height // 2
            draw_star(draw, (cx, cy), outer_r=r, inner_r=r * 0.38, fill=c_star)
            rel_box = to_rel_bbox_1000([cy - r, cx - r, cy + r, cx + r], width, height)
            stars_meta.append({"star_idx": i + 1, "bbox_1000": rel_box})

        delta = actual_count - target_standard_count
        if delta < 0:
            status = "缺少（需补充）"
            action = f"少 {abs(delta)} 颗（需要补充 {abs(delta)} 颗五角星以达到标准 {target_standard_count} 颗）"
        elif delta > 0:
            status = "多增（需删除）"
            action = f"多 {delta} 颗（需要删减 {delta} 颗多余五角星以达到标准 {target_standard_count} 颗）"
        else:
            status = "刚好吻合"
            action = f"数量完全吻合标准 {target_standard_count} 颗，无需多增或少补"

        meta = {
            "task_type": "surplus_and_deficit",
            "standard_target_count": target_standard_count,
            "actual_rendered_count": actual_count,
            "status": status,
            "balance_action": action,
            "delta": delta,
            "stars_bboxes_1000": [s["bbox_1000"] for s in stars_meta]
        }
        return img, meta
