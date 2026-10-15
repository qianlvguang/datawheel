"""基于真实官方标准 SVG 的二创与认知扰动引擎 (Official SVG Remix Engine)。
加载权威官方标准国旗 (ISO 3166-1) 与国徽矢量，并在像素与图层级别施加：
- 真实国旗数星星 (China 5星, Somalia 1星, Vietnam 1星, EU 12星)
- 真实国旗对比图找不同 (消除国徽、缺星、条纹变色)
- 真实国旗视觉加减法 (跨国旗星数加总与差值)
- 真实国旗多增少补 (对照法定标准判定多退少补)
- 官方国徽识别与真实铭文 OCR (西班牙 PLVS VLTRA, 巴西 ORDEM E PROGRESSO)
- 真实国旗空间挖空与完整性破坏
- 真实跨国旗拼接来源识别 (法德拼接、波乌拼接、瑞德拼接)
所有边界框 (BBox) 统一使用以左上角为原点、0~1000 的相对坐标系 [ymin, xmin, ymax, xmax]。
"""
import io
import math
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageDraw
import resvg_py

from .coords import to_rel_bbox_1000
from .palette import COLORS, get_color

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "official_flags"


def load_official_svg_image(code: str, width: int = 640, height: int = 480) -> Image.Image:
    """从本地官方资产库读取 SVG 并通过 resvg 渲染为高质量 PIL 图像。"""
    svg_path = ASSETS_DIR / f"{code.lower()}.svg"
    if not svg_path.exists():
        raise FileNotFoundError(f"未找到官方 SVG 资产: {svg_path}")
    svg_data = svg_path.read_text(encoding="utf-8")
    png_bytes = resvg_py.svg_to_bytes(svg_data, width=width, height=height)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


class OfficialSVGRemixEngine:

    # =========================================================================
    # 1. 真实国旗数星星 (Star Counting on Official Flags)
    # =========================================================================
    @staticmethod
    def task_star_counting(flag_code: str) -> Tuple[Image.Image, Dict[str, Any]]:
        code = flag_code.lower()
        w, h = 640, 480
        img = load_official_svg_image(code, w, h)

        if code == "cn":
            country = "中国国旗 (Five-star Red Flag)"
            star_count = 5
            star_color = "黄色"
            bg_desc = "红色底板，左上角五星拱卫"
            # 真实中国国旗五星的 0~1000 相对坐标 [ymin, xmin, ymax, xmax]
            bboxes = [
                [100, 100, 400, 325],  # 大星
                [80, 360, 180, 435],   # 小星1
                [165, 420, 265, 495],  # 小星2
                [290, 420, 390, 495],  # 小星3
                [385, 360, 485, 435]   # 小星4
            ]
        elif code == "eu":
            country = "欧洲联盟旗帜 (European Union Flag)"
            star_count = 12
            star_color = "金色"
            bg_desc = "蓝色底板，中央十二颗金星环形排列"
            cx, cy = 500, 500
            orbit_r = 300
            star_r = 45
            bboxes = []
            for i in range(12):
                angle = -math.pi / 2 + i * (2 * math.pi / 12)
                sx = cx + orbit_r * math.cos(angle)
                sy = cy + orbit_r * math.sin(angle)
                bboxes.append([int(sy - star_r), int(sx - star_r), int(sy + star_r), int(sx + star_r)])
        elif code == "so":
            country = "索马里国旗 (Somalia Flag)"
            star_count = 1
            star_color = "白色"
            bg_desc = "天蓝色底板，中央一枚白色独立五角星"
            bboxes = [[340, 380, 660, 620]]
        elif code == "vn":
            country = "越南国旗 (Vietnam Flag)"
            star_count = 1
            star_color = "黄色"
            bg_desc = "红色底板，中央一枚大型黄色五角星"
            bboxes = [[300, 350, 700, 650]]
        else:
            country = "官方旗帜"
            star_count = 1
            star_color = "黄色"
            bg_desc = "纯色底板"
            bboxes = [[400, 400, 600, 600]]

        meta = {
            "task_type": "star_counting",
            "country": country,
            "star_count": star_count,
            "star_color": star_color,
            "bg_color": bg_desc,
            "all_bboxes_1000": bboxes,
            "stars_info": [{"star_id": i + 1, "bbox_1000": b} for i, b in enumerate(bboxes)]
        }
        return img, meta

    # =========================================================================
    # 2. 真实国旗对比图找不同 (Spot the Difference on Official Flags)
    # =========================================================================
    @staticmethod
    def task_spot_the_difference(flag_code: str, diff_mode: str) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 480, 360
        img_left = load_official_svg_image(flag_code, w, h)
        img_right = img_left.copy()
        draw_r = ImageDraw.Draw(img_right)

        if flag_code == "cn" and diff_mode == "missing_star":
            # 抹除中国国旗左下角那颗小星 (用红底填补)
            # 对应像素坐标区域
            px_y1, px_x1 = int(0.385 * h), int(0.360 * w)
            px_y2, px_x2 = int(0.485 * h), int(0.435 * w)
            red_fill = img_left.getpixel((10, 10))
            draw_r.rectangle([px_x1, px_y1, px_x2, px_y2], fill=red_fill)
            diff_desc = "右图中，中国国旗四颗小五角星中最下方的一颗被抹除缺失，变成红色底板空白。"
            box_l = [385, 360, 485, 435]
            box_r = [385, 360, 485, 435]
            left_count, right_count = 5, 4

        elif flag_code == "es" and diff_mode == "missing_emblem":
            # 抹除西班牙国旗的官方国徽 (还原为纯黄条纹)
            px_x1, px_x2 = int(0.18 * w), int(0.42 * w)
            px_y1, px_y2 = int(0.28 * h), int(0.72 * h)
            yellow_fill = img_left.getpixel((w // 2, h // 2))
            draw_r.rectangle([px_x1, px_y1, px_x2, px_y2], fill=yellow_fill)
            diff_desc = "右图中，西班牙国旗左侧的官方皇家国徽与大力神柱被完全抹除，变回纯黄色水平横条。"
            box_l = [280, 180, 720, 420]
            box_r = [280, 180, 720, 420]
            left_count, right_count = 1, 0

        elif flag_code == "de" and diff_mode == "color_change":
            # 改变德国国旗最底部的金色横条为浅蓝色
            px_y1 = int(h * 2 / 3)
            draw_r.rectangle([0, px_y1, w, h], fill=(0, 153, 204, 255))
            diff_desc = "右图中，德国国旗最底部的金色横条被篡改为天蓝色横条。"
            box_l = [667, 0, 1000, 1000]
            box_r = [667, 0, 1000, 1000]
            left_count, right_count = 3, 3

        else:
            diff_desc = "右图局部颜色发生了细微改变。"
            box_l = [400, 400, 600, 600]
            box_r = [400, 400, 600, 600]
            left_count, right_count = 1, 1

        # 左右双联画拼接
        total_w = w * 2 + 10
        total_h = h
        composite = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        composite.paste(img_left, (0, 0))
        composite.paste(img_right, (w + 10, 0))
        draw_comp = ImageDraw.Draw(composite)
        draw_comp.rectangle([w, 0, w + 10, total_h], fill=(160, 160, 160, 255))

        # 映射至全图 0~1000 相对坐标
        left_1000 = [
            box_l[0],
            round((box_l[1] * w / 1000) / total_w * 1000),
            box_l[2],
            round((box_l[3] * w / 1000) / total_w * 1000)
        ]
        right_1000 = [
            box_r[0],
            round(((box_r[1] * w / 1000) + w + 10) / total_w * 1000),
            box_r[2],
            round(((box_r[3] * w / 1000) + w + 10) / total_w * 1000)
        ]

        meta = {
            "task_type": "spot_the_difference",
            "diff_type": diff_mode,
            "difference_description": diff_desc,
            "target_differing_box_left_1000": left_1000,
            "target_differing_box_right_1000": right_1000,
            "left_star_count": left_count,
            "right_star_count": right_count
        }
        return composite, meta

    # =========================================================================
    # 3. 真实国旗视觉加减法 (Visual Arithmetic on Real Flags)
    # =========================================================================
    @staticmethod
    def task_visual_arithmetic(flag_a: str, flag_b: str, count_a: int, count_b: int) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 480, 360
        img_a = load_official_svg_image(flag_a, w, h)
        img_b = load_official_svg_image(flag_b, w, h)

        total_w = w * 2 + 10
        total_h = h
        composite = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        composite.paste(img_a, (0, 0))
        composite.paste(img_b, (w + 10, 0))
        draw_comp = ImageDraw.Draw(composite)
        draw_comp.rectangle([w, 0, w + 10, total_h], fill=(160, 160, 160, 255))

        meta = {
            "task_type": "visual_arithmetic",
            "left_flag": flag_a.upper(),
            "right_flag": flag_b.upper(),
            "left_count": count_a,
            "right_count": count_b,
            "sum": count_a + count_b,
            "diff": abs(count_b - count_a),
            "left_bboxes_1000": [[0, 0, 1000, round((w / total_w) * 1000)]],
            "right_bboxes_1000": [[0, round(((w + 10) / total_w) * 1000), 1000, 1000]],
            "arithmetic_expr": f"左侧国旗({flag_a.upper()}，{count_a}颗星) + 右侧国旗({flag_b.upper()}，{count_b}颗星) = 合计 {count_a + count_b} 颗星；差值为 {abs(count_b - count_a)} 颗星。"
        }
        return composite, meta

    # =========================================================================
    # 4. 真实国旗多增少补 (Surplus & Deficit with Real Flags)
    # =========================================================================
    @staticmethod
    def task_surplus_deficit(flag_code: str, target_count: int, actual_count: int) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 640, 480
        img = load_official_svg_image(flag_code, w, h)
        draw = ImageDraw.Draw(img)

        # 如果实际渲染少于标准，抹掉部分元素制造残缺
        delta = actual_count - target_count
        if flag_code == "cn" and actual_count == 3:
            # 抹除中国国旗下面两颗小星
            red_fill = img.getpixel((10, 10))
            draw.rectangle([int(0.35 * w), int(0.28 * h), int(0.51 * w), int(0.50 * h)], fill=red_fill)
            bboxes = [[100, 100, 400, 325], [80, 360, 180, 435], [165, 420, 265, 495]]
            status = "缺少（需补充）"
            action = f"根据中国国旗法定规范应有 5 颗星，当前只有 3 颗星，缺少 2 颗星（需要补充 2 颗小五角星）"

        elif flag_code == "vn" and actual_count == 2:
            # 越南国旗标准 1 星，多画一颗星
            draw_star = False
            # 在右上角额外加一颗星
            from .geometry import draw_star
            draw_star(draw, (int(w * 0.8), int(h * 0.3)), outer_r=h * 0.1, inner_r=h * 0.04, fill=(255, 204, 0))
            bboxes = [[300, 350, 700, 650], [200, 700, 400, 900]]
            status = "多增（需删除）"
            action = f"根据越南国旗法定规范只有 1 颗中心大星，当前画面渲染了 2 颗星，多增 1 颗（需要裁减多余的右上角星）"

        elif flag_code == "so" and actual_count == 1:
            # 吻合
            bboxes = [[340, 380, 660, 620]]
            status = "刚好吻合"
            action = "索马里国旗标准为 1 颗中央白星，当前实际也是 1 颗星，数量完全吻合，无需多增或少补。"

        else:
            bboxes = [[400, 400, 600, 600]]
            status = "缺少（需补充）" if delta < 0 else "多增（需删除）"
            action = f"与法定标准相差 {abs(delta)} 颗符号"

        meta = {
            "task_type": "surplus_and_deficit",
            "country_flag": flag_code.upper(),
            "standard_target_count": target_count,
            "actual_rendered_count": actual_count,
            "delta": delta,
            "status": status,
            "balance_action": action,
            "stars_bboxes_1000": bboxes
        }
        return img, meta

    # =========================================================================
    # 5. 官方国徽识别与真实铭文 OCR (Official Emblem & Text OCR)
    # =========================================================================
    @staticmethod
    def task_official_emblem_ocr(flag_code: str) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 640, 480
        img = load_official_svg_image(flag_code, w, h)

        if flag_code == "es":
            text = "PLVS VLTRA"
            text_desc = "在西班牙官方国徽两侧的大力神柱飘带上，清晰印有拉丁文国家格言'PLVS VLTRA'（意为'海外还有天地'）。"
            emblem_desc = "包含西班牙皇家盾徽（代表卡斯蒂利亚、莱昂、阿拉贡、纳瓦拉和格拉纳达的纹章）及上方皇冠"
            emblem_box = [260, 160, 740, 410]
            text_box = [420, 150, 580, 420]
            layout = "红色、黄色、红色不等宽水平三条纹（中间黄色条纹宽度为上下红色条纹的2倍）"
            colors = ["红色", "黄色"]
        elif flag_code == "br":
            text = "ORDEM E PROGRESSO"
            text_desc = "在巴西国旗中央天球仪的白色拱形弧带上，清晰印有绿色大写葡萄牙文格言'ORDEM E PROGRESSO'（意为'秩序与进步'）。"
            emblem_desc = "包含黄色菱形内部深蓝色天球仪与白色星图弧形飘带"
            emblem_box = [200, 280, 800, 720]
            text_box = [450, 310, 550, 690]
            layout = "绿色旗面底板，中央承载黄色大菱形与深蓝色圆形天球仪"
            colors = ["绿色", "黄色", "深蓝色", "白色"]
        elif flag_code == "mx":
            text = None
            text_desc = "无文字"
            emblem_desc = "墨西哥官方国徽：一只立在仙人掌上的金雕，口中正撕咬着一条响尾蛇，下方环绕橡树和月桂枝叶花环。"
            emblem_box = [300, 360, 700, 640]
            text_box = None
            layout = "绿色、白色、红色三等分垂直竖条纹"
            colors = ["绿色", "白色", "红色", "棕色"]
        else:
            text = None
            text_desc = ""
            emblem_desc = "官方纹章"
            emblem_box = [300, 300, 700, 700]
            text_box = None
            layout = "官方底板"
            colors = ["红色", "白色"]

        meta = {
            "layout_desc": layout,
            "base_colors": colors,
            "element_type": "coat_of_arms",
            "element_color": "复合国徽色",
            "position": "靠左偏中" if flag_code == "es" else "正中央",
            "count": 1,
            "elements_detail": [{"name": emblem_desc, "bbox": emblem_box}],
            "text": text,
            "text_bbox": text_box,
            "text_description": text_desc
        }
        return img, meta

    # =========================================================================
    # 6. 真实国旗空间挖空与完整性破坏 (Real Flag Cutouts & Completeness)
    # =========================================================================
    @staticmethod
    def task_real_flag_cutout(flag_code: str, cutout_type: str) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 640, 480
        img = load_official_svg_image(flag_code, w, h)
        from .modifiers import apply_cutout_or_hollowing
        cut_img, cut_meta = apply_cutout_or_hollowing(img, cutout_type)

        country_map = {"fr": "法国国旗 (蓝白红竖条纹)", "se": "瑞典国旗 (蓝黄北欧十字)", "ua": "乌克兰国旗 (蓝黄横条纹)"}
        meta = {
            "layout_desc": f"{country_map.get(flag_code, '标准国旗')}官方基底",
            "base_colors": ["蓝色", "黄色"] if flag_code in ["se", "ua"] else ["蓝色", "白色", "红色"],
            "element_type": "none",
            "position": "无前景元素",
            "count": 0,
            "elements_detail": []
        }
        meta.update(cut_meta)
        return cut_img, meta

    # =========================================================================
    # 7. 真实跨国旗拼接与溯源 (Real Cross-Country Flag Splicing)
    # =========================================================================
    @staticmethod
    def task_real_flag_splicing(code_a: str, code_b: str, mode: str) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 640, 480
        img_a = load_official_svg_image(code_a, w, h)
        img_b = load_official_svg_image(code_b, w, h)

        name_map = {
            "fr": "法国官方国旗（蓝白红三色竖条纹）",
            "de": "德国官方国旗（黑红金三色横条纹）",
            "pl": "波兰官方国旗（白红二等分横条纹）",
            "ua": "乌克兰官方国旗（蓝黄二等分横条纹）",
            "se": "瑞典官方国旗（蓝底黄色北欧十字）",
            "ch": "瑞士官方国旗（红底白色正十字）"
        }

        meta_a = {"layout_desc": name_map.get(code_a, code_a.upper()), "base_colors": ["色彩A"]}
        meta_b = {"layout_desc": name_map.get(code_b, code_b.upper()), "base_colors": ["色彩B"]}

        from .modifiers import splice_two_flags
        sp_img, sp_meta = splice_two_flags(img_a, meta_a, img_b, meta_b, split_mode=mode)
        return sp_img, sp_meta

    # =========================================================================
    # 8. 真实经典国旗图元、色彩与结构 (Real Standard Flag Base Recognition)
    # =========================================================================
    @staticmethod
    def task_real_standard_flag(flag_code: str) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = 640, 480
        img = load_official_svg_image(flag_code, w, h)

        if flag_code == "ch":
            meta = {
                "layout_desc": "瑞士国旗：鲜红色正方形/矩形底板，正中央配有对称加粗的纯白色希腊正十字",
                "base_colors": ["红色", "白色"],
                "element_type": "cross",
                "element_color": "白色",
                "position": "正中央",
                "count": 1,
                "elements_detail": [{"name": "瑞士白色正十字", "bbox": [300, 350, 700, 650]}]
            }
        elif flag_code == "se":
            meta = {
                "layout_desc": "瑞典国旗：天蓝色基底，正中偏左带有贯穿全旗的黄色北欧十字（十字交点偏向旗杆侧）",
                "base_colors": ["天蓝色", "黄色"],
                "element_type": "nordic_cross",
                "element_color": "黄色",
                "position": "靠旗杆侧（偏左）",
                "count": 1,
                "elements_detail": [{"name": "黄色北欧十字", "bbox": [0, 250, 1000, 450]}]
            }
        elif flag_code == "so":
            meta = {
                "layout_desc": "索马里国旗：轻盈的天蓝色单色底板，正中央嵌有一枚纯白色的独立五角星（代表非洲之角）",
                "base_colors": ["天蓝色", "白色"],
                "element_type": "stars",
                "element_color": "白色",
                "position": "正中央",
                "count": 1,
                "elements_detail": [{"name": "中央白色五角星", "bbox": [340, 380, 660, 620]}]
            }
        else:
            meta = {
                "layout_desc": "标准官方旗帜",
                "base_colors": ["原色"],
                "element_type": "none",
                "position": "居中",
                "count": 0,
                "elements_detail": []
            }
        return img, meta
