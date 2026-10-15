"""多维语义与问答合成器。
基于生成的绝对几何与元数据，同步产出：
1. 细粒度结构化 JSON (Ground Truth)，BBox 统一使用 [0, 1000] 相对坐标系 [ymin, xmin, ymax, xmax]
2. 密集自然语言 Caption
3. 针对 8 大基础能力 + 4 大高阶认知能力（数星星、找不同、视觉加减法、多增少补）的专项微调 QA 对
"""
from typing import Dict, Any, List


class GroundTruthSynthesizer:

    @classmethod
    def build_structured_gt(cls, meta: Dict[str, Any]) -> Dict[str, Any]:
        """产出包含高精度 0~1000 相对坐标的绝对真理结构化 JSON。"""
        task_type = meta.get("task_type")

        # ---------------- 高阶任务 1: 数星星 ----------------
        if task_type == "star_counting":
            return {
                "task": "star_counting",
                "star_count": meta["star_count"],
                "star_color": meta["star_color"],
                "bg_color": meta["bg_color"],
                "coordinate_system": "relative_0_to_1000 [ymin, xmin, ymax, xmax], origin top-left",
                "stars_info": meta["stars_info"],
                "all_bboxes_1000": meta["all_bboxes_1000"]
            }

        # ---------------- 高阶任务 2: 对比图与找不同 ----------------
        if task_type == "spot_the_difference":
            return {
                "task": "spot_the_difference",
                "comparison_layout": "左右双联画对比图（以中间垂直分割线为界，左为原图，右为变化图）",
                "diff_type": meta["diff_type"],
                "difference_description": meta["difference_description"],
                "coordinate_system": "relative_0_to_1000 [ymin, xmin, ymax, xmax], origin top-left",
                "target_differing_box_left_1000": meta["target_differing_box_left_1000"],
                "target_differing_box_right_1000": meta["target_differing_box_right_1000"],
                "left_count": meta["left_star_count"],
                "right_count": meta["right_star_count"]
            }

        # ---------------- 高阶任务 3: 视觉加减法 ----------------
        if task_type == "visual_arithmetic":
            return {
                "task": "visual_arithmetic",
                "arithmetic_expr": meta["arithmetic_expr"],
                "left_count": meta["left_count"],
                "right_count": meta["right_count"],
                "sum": meta["sum"],
                "diff": meta["diff"],
                "coordinate_system": "relative_0_to_1000 [ymin, xmin, ymax, xmax], origin top-left",
                "left_bboxes_1000": meta["left_bboxes_1000"],
                "right_bboxes_1000": meta["right_bboxes_1000"]
            }

        # ---------------- 高阶任务 4: 多增少补 ----------------
        if task_type == "surplus_and_deficit":
            return {
                "task": "surplus_and_deficit",
                "standard_target_count": meta["standard_target_count"],
                "actual_rendered_count": meta["actual_rendered_count"],
                "delta": meta["delta"],
                "status": meta["status"],
                "balance_action": meta["balance_action"],
                "coordinate_system": "relative_0_to_1000 [ymin, xmin, ymax, xmax], origin top-left",
                "stars_bboxes_1000": meta["stars_bboxes_1000"]
            }

        # ---------------- 基础任务: 旗帜几何图元 ----------------
        base_colors = meta.get("base_colors", [])
        elem_color = meta.get("element_color")
        all_colors = list(dict.fromkeys(base_colors + ([elem_color] if elem_color else [])))

        elem_type = meta.get("element_type", "none")
        has_emblem = elem_type == "coat_of_arms"
        is_complete = meta.get("is_complete", True)

        is_spliced = meta.get("is_spliced", False)
        if is_spliced and "source_a" in meta and "source_b" in meta:
            sa = meta["source_a"]
            sb = meta["source_b"]
            base_colors = list(dict.fromkeys(sa.get("base_colors", []) + sb.get("base_colors", [])))
            all_colors = base_colors
            layout_desc = meta.get("splice_desc")
        else:
            layout_desc = meta.get("layout_desc", meta.get("layout", "unknown"))

        return {
            "coordinate_system": "relative_0_to_1000 [ymin, xmin, ymax, xmax], origin top-left",
            "layout_and_topology": layout_desc,
            "colors": {
                "all_colors": all_colors,
                "base_stripes_or_field": base_colors,
                "symbols_color": elem_color
            },
            "elements_detected": {
                "category": elem_type,
                "symbols_list": [e["name"] for e in meta.get("elements_detail", [])]
            },
            "emblem_details": {
                "has_emblem": has_emblem,
                "emblem_description": "包含盾徽与铭文飘带" if has_emblem else "无独立国徽"
            },
            "spatial_orientation": {
                "element_position": meta.get("position", "无独立前景元素"),
                "coordinates_bbox_1000": [e.get("bbox") for e in meta.get("elements_detail", [])]
            },
            "completeness_and_cutout": {
                "is_complete": is_complete,
                "cutout_details": meta.get("cutout_desc", "图案完整无缺损"),
                "cutout_bbox_1000": meta.get("cutout_bbox")
            },
            "quantity_and_counting": {
                "star_or_symbol_count": meta.get("count", 0),
                "rays_count": meta.get("rays_count", 0),
                "detail": f"共包含 {meta.get('count', 0)} 个前景主元素"
            },
            "splicing_composition": {
                "is_spliced": is_spliced,
                "splice_description": meta.get("splice_desc", "单张完整旗帜，未经过拼接合成")
            },
            "ocr_text": {
                "text_found": meta.get("text") is not None,
                "text_content": meta.get("text"),
                "text_bbox_1000": meta.get("text_bbox")
            }
        }

    @classmethod
    def build_dense_caption(cls, meta: Dict[str, Any], gt: Dict[str, Any]) -> str:
        """生成密集自然语言段落描述。"""
        task_type = meta.get("task_type")

        if task_type == "star_counting":
            return (f"该图像在{meta['bg_color']}背景下分布着共计 {meta['star_count']} 颗{meta['star_color']}五角星。"
                    f"所有五角星各自分散排布，彼此无重叠，呈现出点阵分布特征。")

        if task_type == "spot_the_difference":
            return (f"这是一幅左右并列的双联画对比图。左图为标准基准图，右图为变动图。"
                    f"核心差异为：{meta['difference_description']}。两图在其余位置均保持严格对称一致。")

        if task_type == "visual_arithmetic":
            return (f"这是一幅视觉算术对比图。左侧画板包含 {meta['left_count']} 颗黄色五角星，"
                    f"右侧画板包含 {meta['right_count']} 颗黄色五角星。"
                    f"两幅画板合计共有 {meta['sum']} 颗星星，数量差值为 {meta['diff']} 颗。")

        if task_type == "surplus_and_deficit":
            return (f"该图案作为旗帜基底，依照特定标准原本应包含 {meta['standard_target_count']} 颗星，"
                    f"但当前实际只渲染了 {meta['actual_rendered_count']} 颗星。"
                    f"当前状态为[{meta['status']}]，需要执行调整：{meta['balance_action']}。")

        # 默认基础旗帜描述
        parts = []
        if gt["splicing_composition"]["is_spliced"]:
            parts.append(gt["splicing_composition"]["splice_description"])
        else:
            parts.append(f"该图像展示了一面标准几何旗帜。底板采用{gt['layout_and_topology']}，"
                         f"基底主要包含色彩：{', '.join(gt['colors']['base_stripes_or_field'])}。")

        elem_cat = gt["elements_detected"]["category"]
        if elem_cat != "none":
            elem_names = ", ".join(gt["elements_detected"]["symbols_list"])
            parts.append(f"在画面的{gt['spatial_orientation']['element_position']}，"
                         f"绘有{gt['colors']['symbols_color']}的几何符号组合（包含{elem_names}），"
                         f"数量共计{gt['quantity_and_counting']['star_or_symbol_count']}个。")

        if gt["emblem_details"]["has_emblem"]:
            text_info = f"，下方铭文飘带中清晰印有文字'{gt['ocr_text']['text_content']}'" if gt["ocr_text"]["text_found"] else ""
            parts.append(f"中心承载着一枚庄严的纹章盾徽{text_info}。")

        if not gt["completeness_and_cutout"]["is_complete"]:
            parts.append(f"【注意完整性异常】：{gt['completeness_and_cutout']['cutout_details']}。")
        else:
            parts.append("整张旗帜边缘规整，图案结构完整。")

        return "".join(parts)

    @classmethod
    def build_qa_dataset(cls, meta: Dict[str, Any], gt: Dict[str, Any]) -> List[Dict[str, str]]:
        """针对不同任务维度，生成高精度的多轮微调问答对。"""
        task_type = meta.get("task_type")
        qas = []

        # ---------------- 高阶任务 1: 数星星 ----------------
        if task_type == "star_counting":
            qas.append({
                "dimension": "counting",
                "question": "请仔细清点画面中一共有多少颗五角星？",
                "answer": f"经过逐一清点，画面中共有 {meta['star_count']} 颗五角星。"
            })
            qas.append({
                "dimension": "grounding",
                "question": "请列出画面中每一颗五角星在 0~1000 相对坐标系下的定位框 [ymin, xmin, ymax, xmax]。",
                "answer": f"所有五角星在 0~1000 相对坐标系下的边界框如下：{meta['all_bboxes_1000']}。"
            })
            return qas

        # ---------------- 高阶任务 2: 对比图与找不同 ----------------
        if task_type == "spot_the_difference":
            qas.append({
                "dimension": "spot_difference",
                "question": "这是一幅左右对比图，请找出左图与右图之间存在的核心差异是什么？",
                "answer": meta["difference_description"]
            })
            qas.append({
                "dimension": "difference_localization",
                "question": "请使用 0~1000 相对坐标系 [ymin, xmin, ymax, xmax]，分别指出左图与右图中发生差异的对应区域。",
                "answer": (f"在 0~1000 归一化相对坐标系中：\n"
                           f"- 左图中原位置坐标为：{meta['target_differing_box_left_1000']}\n"
                           f"- 右图中变动/缺失位置坐标为：{meta['target_differing_box_right_1000']}")
            })
            qas.append({
                "dimension": "quantity_comparison",
                "question": "左图和右图的五角星数量各是多少颗？",
                "answer": f"左图包含 {meta['left_star_count']} 颗五角星，右图包含 {meta['right_star_count']} 颗五角星。"
            })
            return qas

        # ---------------- 高阶任务 3: 视觉加减法 ----------------
        if task_type == "visual_arithmetic":
            qas.append({
                "dimension": "visual_addition",
                "question": "左侧画板有几颗星？右侧画板有几颗星？两幅图合计共有几颗星？",
                "answer": (f"左侧画板有 {meta['left_count']} 颗星，右侧画板有 {meta['right_count']} 颗星。"
                           f"两边加总合计共有 {meta['sum']} 颗星（{meta['left_count']} + {meta['right_count']} = {meta['sum']}）。")
            })
            qas.append({
                "dimension": "visual_subtraction",
                "question": "右侧画板与左侧画板的星星数量差值是多少？",
                "answer": f"数量差值为 {meta['diff']} 颗（|{meta['right_count']} - {meta['left_count']}| = {meta['diff']}）。"
            })
            return qas

        # ---------------- 高阶任务 4: 多增少补 ----------------
        if task_type == "surplus_and_deficit":
            qas.append({
                "dimension": "target_balancing_status",
                "question": (f"依照既定标准，该图案理应拥有 {meta['standard_target_count']} 颗五角星。"
                             f"请对照画面中的实际数量，判断当前是'多增'、'缺少'还是'吻合'？"),
                "answer": f"当前实际数量为 {meta['actual_rendered_count']} 颗，判定状态为：{meta['status']}。"
            })
            qas.append({
                "dimension": "balance_action",
                "question": "为了恢复为标准数量，具体应采取什么增删调整行动？",
                "answer": f"调整方案：{meta['balance_action']}。"
            })
            qas.append({
                "dimension": "current_bboxes",
                "question": "请列出当前已有星星在 0~1000 相对坐标系下的位置。",
                "answer": f"当前各星边界框为：{meta['stars_bboxes_1000']}。"
            })
            return qas

        # ---------------- 基础任务问答 ----------------
        qas.append({
            "dimension": "color",
            "question": "请详细列出画面中所有出现的颜色，并说明各自属于哪部分？",
            "answer": (f"画面中出现的颜色包括：{', '.join(gt['colors']['all_colors'])}。"
                       f"其中底板使用了{', '.join(gt['colors']['base_stripes_or_field'])}；"
                       f"前景符号与图案使用了{gt['colors']['symbols_color'] or '无前景元素'}。")
        })

        qas.append({
            "dimension": "element",
            "question": "画面中包含哪些具体的几何或纹章象征元素？",
            "answer": (f"画面中识别出的主要元素类别为[{gt['elements_detected']['category']}]，"
                       f"具体包含以下图元：{', '.join(gt['elements_detected']['symbols_list']) or '纯底板无额外图元'}。")
        })

        qas.append({
            "dimension": "emblem",
            "question": "图像中是否存在国徽或纹章？如果有，请描述其特征。",
            "answer": ("画面中包含一枚纹章盾徽，具有明显的盾形边框与内部象征图腾。"
                       if gt["emblem_details"]["has_emblem"] else "画面中没有独立的国徽或纹章盾牌，仅为常规几何图元。")
        })

        qas.append({
            "dimension": "orientation",
            "question": "核心图元或徽章位于旗帜的哪个具体方位？并给出 0~1000 相对坐标。",
            "answer": f"核心图元位于旗帜的{gt['spatial_orientation']['element_position']}，相对边界框为：{gt['spatial_orientation']['coordinates_bbox_1000']}。"
        })

        qas.append({
            "dimension": "completeness",
            "question": "请检查这面旗帜的完整性，是否存在挖空、遮挡或残损区域？",
            "answer": ("该图案不完整！" + gt["completeness_and_cutout"]["cutout_details"] + f"，缺损坐标为：{gt['completeness_and_cutout']['cutout_bbox_1000']}"
                       if not gt["completeness_and_cutout"]["is_complete"] else "该图案非常完整，不存在任何挖空、镂空或缺角破损。")
        })

        qas.append({
            "dimension": "quantity",
            "question": "画面中特定符号（如星星、光芒等）的精确数量是多少？",
            "answer": gt["quantity_and_counting"]["detail"] + (f"，光芒尖刺数量为 {gt['quantity_and_counting']['rays_count']}" if gt['quantity_and_counting']['rays_count'] > 0 else "") + "。"
        })

        qas.append({
            "dimension": "splicing",
            "question": "这是一面单一独立的旗帜，还是经过了拼贴组合？请说明其来源结构。",
            "answer": gt["splicing_composition"]["splice_description"]
        })

        qas.append({
            "dimension": "ocr",
            "question": "图案中是否存在可识别的文字或铭文？如果有，内容是什么？坐标在哪里？",
            "answer": (f"存在文字，在铭文飘带中识别出文字内容为：'{gt['ocr_text']['text_content']}'，位于 0~1000 坐标：{gt['ocr_text']['text_bbox_1000']}。"
                       if gt["ocr_text"]["text_found"] else "画面中纯为几何图案与色块，未检测到任何文本或文字铭文。")
        })

        return qas
