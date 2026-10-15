"""多模态中训练问答与监督协议合成器 (Reference QA Synthesizer)。
支持以下主流大模型中训练格式：
1. In-Context 多图参考检索协议 (Reference Image vs Query Image Grounding)
2. 细粒度候选集辨析协议 (Multiple Candidates Disambiguation)
3. 显式不确定性与抗幻觉拒绝协议 (Evidence Calibration & Refusal Protocol)
4. 标准化 0~1000 相对坐标标注: [ymin, xmin, ymax, xmax]
"""
from typing import Dict, Any, List, Optional
import json


class ReferenceQASynthesizer:
    """构建多模态中训练对话与监督 Ground Truth"""

    @classmethod
    def build_in_context_qa(cls, sample: Dict[str, Any],
                            ref_card_path: str,
                            query_img_path: str) -> Dict[str, Any]:
        """构建两张图像输入（参考卡片 + 商品实物图）的参考匹配样本。"""
        target_name = sample["target_name"]
        match_status = sample["match_status"]
        bbox = sample["bbox_1000"]
        reason = sample["grounding_reason"]

        user_prompt = (
            f"请比对参考图像 <|ref_image|> 与商品实拍图 <|query_image|>。\n"
            f"任务要求：\n"
            f"1. 在商品图中定位是否存在与参考目标【{target_name}】相符合的印花或图案，定位框使用以左上角为原点的 0~1000 相对坐标 [ymin, xmin, ymax, xmax]；\n"
            f"2. 判断匹配状态：exact_match (完全匹配), variant_match (透视/材质变体), robust_match (轻度磨损模糊但可辨), "
            f"hard_mismatch_distractor (相似但非该目标), insufficient_evidence (证据不足/严重遮挡无法断言), absent (无目标)；\n"
            f"3. 简述判定证据与细节理由。"
        )

        response_payload = {
            "target": target_name,
            "match_status": match_status,
            "bbox_1000": bbox,
            "recognizability": sample["recognizability"],
            "rationale": reason
        }

        # 转换为主流 VLM 易读的结构化 Markdown + JSON 文本
        if bbox is not None:
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        else:
            bbox_str = "null"

        assistant_response = (
            f"```json\n"
            f"{json.dumps(response_payload, ensure_ascii=False, indent=2)}\n"
            f"```\n\n"
            f"**定位与判定总结**：\n"
            f"- 目标实体：{target_name}\n"
            f"- 定位坐标：`{bbox_str}`\n"
            f"- 匹配判定：`{match_status}`\n"
            f"- 视觉依据：{reason}"
        )

        return {
            "conversation_type": "in_context_reference_matching",
            "state_key": sample["state_key"],
            "ref_image": ref_card_path,
            "query_image": query_img_path,
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "target_ref_id": sample["target_ref_id"],
                "target_name": target_name,
                "match_status": match_status,
                "bbox_1000": bbox,
                "confidence": sample["confidence"]
            }
        }

    @classmethod
    def build_candidate_disambiguation_qa(cls, sample: Dict[str, Any],
                                          candidate_names: List[str],
                                          query_img_path: str) -> Dict[str, Any]:
        """构建候选集多选一辨析样本（包含困难干扰项对比）。"""
        target_name = sample["target_name"]
        match_status = sample["match_status"]
        bbox = sample["bbox_1000"]
        reason = sample["grounding_reason"]

        cands_str = "、".join([f"【{c}】" for c in candidate_names])

        user_prompt = (
            f"在商品实物图 <|query_image|> 中，可能出现了以下候选图案之一：{cands_str}，或者不存在上述图案。\n"
            f"请分析商品图中的印花属于哪一个？给出检测框 [ymin, xmin, ymax, xmax] 及细粒度甄别理由。"
        )

        if match_status in ["exact_match", "variant_match", "robust_match"]:
            decision = f"该商品图案匹配候选中的【{target_name}】。"
        elif match_status == "hard_mismatch_distractor":
            distractor_name = sample.get("distractor_name", "相似干扰项")
            decision = f"该商品图案实际为【{distractor_name}】，排除了【{target_name}】。"
        elif match_status == "insufficient_evidence":
            decision = "图像存在严重退化或遮挡，关键拓扑不可见，证据不足以断言属于任一指定候选。"
        else:
            decision = "商品图中未检出上述任何候选图案。"

        assistant_response = f"{decision}\n定位边界框：`{bbox}`\n分析理由：{reason}"

        return {
            "conversation_type": "candidate_disambiguation",
            "query_image": query_img_path,
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
