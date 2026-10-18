#!/usr/bin/env python3
"""清理 Datawheel 数据文本中的实现术语/图形学术语。

目标：让 caption / QA / 结构化输出更像真实世界的"物品 + 名称 + 可见特征"描述，
不暴露数据生成实现细节（如程序化、矢量、模式名）或训练元话语（"本样本训练模型…"）。

处理对象：
1. v2 分片 JSONL（output/datawheel_v2/_generation/shard_records/*.jsonl）
   —— 处理后需重新执行 generate_datawheel_v2.py --merge 生成 v2 parquet/report；
2. V6 parts parquet（pilot / balanced）—— 处理后需重新 pack / assemble；
3. 可选：任意含 messages / caption 列的 parquet（--extra-dataset 传入）。

规则全部为确定性替换，可重复执行（幂等）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_SHARDS = PROJECT_ROOT / "output" / "datawheel_v2" / "_generation" / "shard_records"
V6_PARTS = [
    PROJECT_ROOT / "output" / "datawheel_v6_pilot" / "parts" / "single_image.parquet",
    PROJECT_ROOT / "output" / "datawheel_v6_pilot" / "parts" / "reference_multi.parquet",
    PROJECT_ROOT / "output" / "datawheel_v6_balanced" / "parts" / "single_rows.parquet",
    PROJECT_ROOT / "output" / "datawheel_v6_balanced" / "parts" / "reference_rows.parquet",
]

MODE_ZH = {
    "tricolor_vertical": "竖向色块区域",
    "tricolor_horizontal": "横向色块区域",
    "quadrant_grid": "四象限色块区域",
    "centered_disc": "中央圆形图案区域",
}

ZONE_ZH = {
    "center_chest_graphic": "前胸中央印花区",
    "left_chest_badge": "左胸上方徽标区",
    "lower_hem_patch": "下摆边缘标牌区",
    "bag_front_flap": "包盖/正面外袋装饰区",
    "bag_front_panel": "包身正面印花区",
    "shoe_lateral_quarter": "鞋面/鞋帮标识区",
    "shoe_tongue_panel": "鞋舌/鞋面小标区",
    "hat_front_panel": "帽冠正面区域",
    "thigh_side_graphic": "裤腿印花区",
    "pants_hem_patch": "裤脚口标牌区",
    "bodice_center_graphic": "裙身上半部印花区",
    "skirt_hem_patch": "裙摆边缘标牌区",
    "full_canvas_composition": "整幅画面",
    "side_by_side_reference_and_packshot": "参考图与商品图并排区域",
}
CATEGORY_MAP = {
    "男士上装/T恤与卫衣": "男士T恤/卫衣",
    "女士时尚箱包/双肩包与手提包": "女士包袋",
    "男女同款/中性时尚箱包/双肩包与手提包": "包袋",
    "男士鞋履/运动休闲鞋": "男士运动鞋",
    "男女同款/中性帽饰/棒球帽与针织帽": "棒球帽/针织帽",
    "女士上装/T恤与卫衣": "女士T恤/卫衣",
    "男士帽饰/棒球帽与针织帽": "男士棒球帽/针织帽",
    "真人穿搭/秀场实拍时装连衣裙": "连衣裙",
    "女士下装/休闲裤与牛仔裤": "女士休闲裤/牛仔裤",
    "男士下装/休闲裤与牛仔裤": "男士休闲裤/牛仔裤",
    "女士连衣裙/一体式时装": "女士连衣裙",
    "男士时尚箱包/双肩包与手提包": "男士包袋",
    "女士鞋履/运动休闲鞋": "女士运动鞋",
    "女士帽饰/棒球帽与针织帽": "女士棒球帽/针织帽",
    "真人穿搭/秀场实拍长款大衣/风衣": "长款大衣/风衣",
    "真人穿搭/秀场实拍T恤/卫衣上装": "T恤/卫衣",
    "真人穿搭/秀场实拍衬衫/罩衫时装": "衬衫/罩衫",
    "男童/青少年上装/T恤与卫衣": "男童T恤/卫衣",
    "女童/青少年连衣裙/一体式时装": "女童连衣裙",
    "女童/青少年下装/休闲裤与牛仔裤": "女童休闲裤/牛仔裤",
    "真人穿搭/秀场实拍夹克/机能外套": "夹克/机能外套",
    "男女同款/中性鞋履/运动休闲鞋": "运动鞋",
    "女童/青少年上装/T恤与卫衣": "女童T恤/卫衣",
    "男女同款/中性上装/T恤与卫衣": "中性T恤/卫衣",
    "真人穿搭/秀场实拍针织毛衣/羊毛衫": "针织毛衣/羊毛衫",
    "男童/青少年鞋履/运动休闲鞋": "男童运动鞋",
    "真人穿搭/秀场实拍开衫外套": "开衫外套",
    "纯色平面画布": "纯色背景",
}
SURFACE_MAP = {
    "繁复印花/强纹理面料表面": "印花较复杂、面料纹理明显",
    "深色纯色面料表面": "深色面料",
    "浅色纯色面料表面": "浅色面料",
    "细腻图案/暗纹面料表面": "带细腻暗纹的面料",
    "自然褶皱与光影渐变": "面料有自然褶皱",
}
TECHNIQUE_MAP = {
    "高精度工业丝印胶印": "丝网印刷",
    "高精度工业丝网胶印": "丝网印刷",
    "精细电脑刺绣": "电脑刺绣",
    "织唛/织标": "织唛",
    "高精度工业丝印": "丝网印刷",
    "高精度热转印": "热转印",
    "高精度数码直喷": "数码直喷",
    "高精度刺绣贴布": "刺绣",
    "高精度塑胶浆": "胶浆印花",
}
GEOMETRY_MAP = {
    "标准 4:3 / 3:2 传统旗帜与徽章比例": "长宽比约 4:3",
    "标准 1:1 圆形徽章比例": "接近圆形",
    "正方形/对称正圆同心圆构型": "接近正方形或圆形",
    "横向宽幅扁平矩形构型": "横向长方形",
    "纵向立式纤细长条构型": "纵向长条形",
}
SCENE_MAP = {
    "电商标准影棚实拍白底/净底商品图": "纯白背景",
    "纯色平面背景": "纯色背景",
    "高分辨率街拍/时装秀场模特实景穿搭图": "街拍/秀场场景",
}
POSITION_TOKENS = ["左上", "右上", "左下", "右下", "顶层", "中层", "底层", "左侧", "右侧", "中间", "正中", "全幅"]
POSITION_ZH = {
    "左上": "左上", "右上": "右上", "左下": "左下", "右下": "右下",
    "顶层": "顶层", "中层": "中层", "底层": "底层",
    "左侧": "左侧", "右侧": "右侧", "中间": "中间", "正中": "中央", "全幅": "背景",
}
SHAPE_TOKENS = ["竖向色块", "水平条带", "方形分区", "圆形徽记", "底板"]
SHAPE_EN = {
    "竖向色块": "vertical panel", "水平条带": "horizontal stripe",
    "方形分区": "quadrant panel", "圆形徽记": "circular emblem", "底板": "background",
}
VISUAL_REPLACEMENTS = [
    (re.compile(r"候选数:\s*(\d+);\s*匹配标签:\s*match;.*", re.S), r"图中有 \1 张候选参考图，商品上的图案与其中一张一致。"),
    (re.compile(r"候选数:\s*(\d+);\s*匹配标签:\s*no_match;.*", re.S), r"图中有 \1 张候选参考图，均与商品上的图案不一致。"),
    (re.compile(r"候选数:\s*(\d+);\s*匹配标签:\s*insufficient_evidence;.*", re.S), r"图中有 \1 张候选参考图，均无法与商品图案建立确认匹配。"),
    (re.compile(r"样本类型:\s*occluded;.*", re.S), "目标区域被遮挡，图案不可见。"),
    (re.compile(r"样本类型:\s*hard_distractor;.*", re.S), "图中出现了外形相似但不同的图案。"),
    (re.compile(r"样本类型:\s*absent;.*", re.S), "图中没有出现目标图案。"),
    (re.compile(r"模式:\s*[^;]+;\s*分区数:\s*(\d+);\s*边界锐利、无物理形变。", re.S), r"画面由 \1 个颜色区域组成，色块边界清晰。"),
    (re.compile(r"布局:\s*([^;]+);\s*图元数:\s*(\d+);\s*空间关系:\s*([^。]+)。"), r"画面采用\1布局，共有 \2 个图案；\3。"),
    (re.compile(r"面料:\s*([^;]+);\s*工艺:\s*([^;]+);\s*印花可见度:\s*[\d.]+。"), r"面料为\1，采用\2，图案清晰可见。"),
]
TEXT_REPLACEMENTS = [
    # 结构/元信息
    ("程序化几何基准样本", "平面图案"),
    ("程序化高精度平整色块底板", "纯色平面背景"),
    ("程序化几何画布", "纯色平面画布"),
    ("所有分区边界笔直锐利、色彩均匀，无纹理与透视干扰。", "各颜色区域边界清晰，整体是一幅平面图案。"),
    ("边界笔直锐利、色彩均匀，无纹理与透视干扰。", "颜色区域边界清晰。"),
    ("边界锐利、无物理形变。", "颜色区域边界清晰。"),
    ("无纹理与透视干扰。", ""),
    ("无物理形变。", ""),
    ("纯色几何分区", "纯色区域"),
    ("几何色彩分区", "颜色区域"),
    ("几何分区", "颜色区域"),
    ("程序化", ""),
    ("合成样本", "图像"),
    ("样本类型:", ""),
    # 图形学/实现术语
    ("典藏官方矢量徽标", "典藏官方徽标"),
    ("官方矢量徽标", "官方徽标"),
    ("矢量徽标", "徽标"),
    ("二维矢量封闭路径", "清晰的外形轮廓"),
    ("矢量图形", "图案"),
    ("矢量图", "图案"),
    ("矢量", ""),
    ("标准无畸变基准模板", "清晰的同款参考图"),
    ("无畸变基准模板", "清晰的同款参考图"),
    ("基准模板", "参考图"),
    ("标准 RGB 平面位图", "参考图"),
    ("RGB 平面位图", "参考图"),
    ("平面位图", "图像"),
    ("位图", "图像"),
    ("归一化边界框", "边界框"),
    ("归一化定位边界框", "边界框"),
    ("归一化坐标", "坐标"),
    ("归一化", ""),
    ("跨模态", ""),
    ("跨域", ""),
    ("恒常性", "一致性"),
    ("构图解析", "图案特征"),
    ("物理贴合呈现", "印花细节"),
    ("可见 Alpha 掩模", "可见区域"),
    ("Alpha 掩模", "可见区域"),
    ("Alpha 外接框", "可见边界"),
    ("外接框", "边界框"),
    ("掩模", "可见区域"),
    ("有效像素的中位色", "主体区域的主要颜色"),
    ("中位色", "主要颜色"),
    ("像素", "区域"),
    ("图元组合", "图案组合"),
    ("图元", "图案"),
    ("形变后的", "贴合后的"),
    # 阶段标题 / 研究训练元话语
    ("阶段A：基础色块与几何图案", "阶段A：基础色块与颜色区域"),
    ("阶段B：真实商品单图案物理贴合", "阶段B：真实商品单图案展示"),
    ("阶段C：真实商品多图案空间排布", "阶段C：真实商品多图案组合"),
    ("面料有自然褶皱的面料表面", "面料有自然褶皱"),
    ("的面料表面的", "的"),
    ("的表面的", "的"),
    ("物理贴合", ""),
    ("商品正面可视面的中部居中区域", "正面中央"),
    ("正面可视面的", "正面"),
    ("可视面", ""),
    ("高精度工业丝印胶印", "丝网印刷"),
    ("高精度工业丝印", "丝网印刷"),
    ("高精度热转印", "热转印"),
    ("高精度数码直喷", "数码直喷"),
    ("高精度刺绣贴布", "刺绣"),
    ("高精度塑胶浆", "胶浆印花"),
    ("与底布形成自然的丝印/热转印质感", "印花与面料融合自然"),
    ("与底布形成的丝印/热转印质感", "印花与面料融合自然"),
    ("同色面料/遮挡物", "布料"),
    ("未暴露可辨识的图案", "看不到图案"),
    ("标准 4:3 / 3:2 传统旗帜与徽章比例", "长宽比约 4:3"),
    ("外形呈长宽比约", "整体长宽比约"),
    ("标准 1:1 圆形徽章比例", "接近圆形"),
    ("与底布形成的丝印/热转印质感", "与底布形成自然的丝印/热转印质感"),
    ("边缘随面料褶皱产生自然形变", "边缘随面料褶皱起伏"),
    ("随面料褶皱产生形变", "随面料褶皱自然起伏"),
    ("各图案均随", "各图案随"),
    ("几何色块分割", "色块搭配"),
    ("反算", "确定"),
    ("阶段D：参考图匹配与对照", "阶段D：参考图匹配与实物对照"),
    ("模型在无畸变参考图与复杂真实商品之间的身份匹配能力。", ""),
    ("无畸变参考图", "参考图"),
    ("复杂真实商品", "实物商品"),
    ("候选框", "候选图中的图案框"),
    ("商品实例框", "商品图案框"),
    ("参考图匹配对照样本：", "参考图与商品对照："),
    ("本样本用于训练模型在证据不足时拒绝臆断，而不是凭先验回答。", "该区域没有可见图案，无法进一步确认。"),
    ("本负样本训练模型正确否定不存在的目标，抑制误报。", "画面中没有出现该图案。"),
    ("该样本用于训练模型抵抗相似图元干扰，避免仅凭粗略外观误判。", "两处图案外形相近，需要仔细区分。"),
    ("该任务检验模型在无畸变参考图与复杂真实商品之间的跨域身份匹配能力。", "可用于核对同一图案在参考图与实物商品上的一致性。"),
    ("验证模型在无畸变设计图与复杂物理实物之间的跨域特征恒常性检索能力。", "可用于对照同一图案在参考图与实物商品上的一致性。"),
    ("本样本为跨模态图文检索与实例比对测试集：", ""),
    ("本样本为", ""),
    ("本负样本", ""),
    ("本样本", ""),
    ("该样本用于", ""),
    ("该任务检验", ""),
    # 英文
    ("Procedural geometric baseline", "A flat color-block graphic"),
    ("procedural geometric baseline", "a flat color-block graphic"),
    ("In-context reference matching sample", "Reference matching pair"),
    ("cross-domain identity matching", "reference-to-product matching"),
    ("This tests cross-domain", "This checks whether the same pattern appears in both places; previously it tested cross-domain"),
    ("This sample trains calibrated rejection under insufficient evidence.", "The region shows no visible pattern, so the answer cannot be confirmed."),
    ("This hard negative trains fine-grained discrimination.", "The two patterns look similar and require careful discrimination."),
    ("This negative sample trains correct rejection and reduces false positives.", "There is no such pattern in the image."),
    ("normalized bounding box", "bounding box"),
    ("vector logo", "logo"),
    ("vector graphic", "graphic"),
    ("vector", ""),
]
META_SENTENCE_RE = re.compile(r"(本样本|本负样本|该样本用于|该任务检验)[^。！？]*[。！？]?")


def normalize_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    text = value
    # stage A caption: 去掉实现模式前缀（程序化几何基准样本 / 平面图案（mode））
    text = re.sub(r"^(?:程序化几何基准样本|平面图案)（[^）]*）：", "", text)
    text = re.sub(r"^(?:Procedural geometric baseline|A flat color-block graphic|A flat graphic) \([^)]*\) with ",
                  "A flat graphic with ", text)
    for code, zh in {**MODE_ZH, **ZONE_ZH}.items():
        text = text.replace(f'"placement_zone": "{code}"', f'"placement_zone": "{zh}"')
        text = text.replace(f"placement_zone={code}", f"placement_zone={zh}")
    for old, new in {**CATEGORY_MAP, **SCENE_MAP, **SURFACE_MAP, **TECHNIQUE_MAP, **GEOMETRY_MAP}.items():
        text = text.replace(old, new)
    for pattern, repl in VISUAL_REPLACEMENTS:
        text = pattern.sub(repl, text)
    for old, new in TEXT_REPLACEMENTS:
        if old:
            text = text.replace(old, new)
    text = re.sub(r"主图(?!案)", "主图案", text)
    text = META_SENTENCE_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"：\s*。", "。", text)
    text = re.sub(r"；\s*。", "。", text)
    text = text.strip()
    return text if text else value


def rewrite_stage_a_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Stage A: 去掉"区域名里带颜色 + 又描述颜色"的重复，以及 caption 里的 bbox/模式信息。"""
    caption = row.get("caption_zh", "")
    if "的颜色为" not in caption:
        return row
    try:
        entities = json.loads(row.get("entities_json") or "[]")
    except Exception:
        return row
    if not entities:
        return row
    old_to_new: Dict[str, str] = {}
    descriptors: List[str] = []
    for ent in entities:
        name = ent.get("entity_name_zh", "")
        pos = next((p for p in POSITION_TOKENS if name.startswith(p)), "")
        shape = next((sh for sh in SHAPE_TOKENS if name.endswith(sh)), "色块")
        color = name[len(pos):]
        if color.endswith(shape):
            color = color[: -len(shape)]
        color = color.strip(" /")
        pos_label = POSITION_ZH.get(pos, pos)
        if pos == "全幅":
            new_name = "背景"
            en_name = "background"
        elif pos == "正中":
            new_name = "中央" + shape
            en_name = "centered " + SHAPE_EN.get(shape, "region")
        else:
            new_name = pos + shape
            en_name = SHAPE_EN.get(shape, "region")
        old_to_new[name] = new_name
        ent["entity_name_zh"] = new_name
        ent["entity_name_en"] = en_name
        if color:
            descriptors.append(f"{pos_label}为{color}")
    n = len(entities)
    row["caption_zh"] = f"画面由 {n} 个纯色区域组成：" + "，".join(descriptors) + "。"
    row["caption_en"] = f"A flat graphic with {n} solid-color regions."
    row["visual_analysis"] = f"画面由 {n} 个颜色区域组成。"
    row["entities_json"] = json.dumps(entities, ensure_ascii=False)
    row["entity_names_zh"] = " / ".join(e["entity_name_zh"] for e in entities)
    row["entity_names_en"] = " / ".join(e["entity_name_en"] for e in entities)
    qa = row.get("qa_pairs_json", "")
    for old, new in old_to_new.items():
        qa = qa.replace(old, new)
    qa = qa.replace("颜色分区", "颜色区域").replace("独立分区", "独立颜色区域")
    qa = qa.replace("按色块边界线统计封闭区域数量。", "按颜色区域的边界统计。")
    qa = qa.replace("直接读取该色块的四条边界。", "读取该颜色区域的边界。")
    qa = qa.replace("该分区为单一纯色填充。", "该区域为单一纯色。")
    row["qa_pairs_json"] = qa
    return row


def clean_free_caption(cap: str) -> str:
    """对任意 caption 文本做自然语言清理（不依赖行级字段）。"""
    if not cap:
        return cap
    for old, new in {**CATEGORY_MAP, **SCENE_MAP, **SURFACE_MAP, **TECHNIQUE_MAP, **GEOMETRY_MAP}.items():
        cap = cap.replace(old, new)
    cap = re.sub(r"图像为一件(.+?)实拍商品（[^）]*）", r"图中是一件\1", cap)
    cap = re.sub(r"图像为一件(.+?)实拍商品", r"图中是一件\1", cap)
    cap = re.sub(r"图中是一件(.+?)商品", r"图中是一件\1", cap)
    cap = re.sub(r"图案特征：[^。]*。", "", cap)
    cap = re.sub(r"[，；]?该图案在整图中的边界框为\s*\[[^\]]*\]。", "", cap)
    cap = re.sub(r"[，；]?边界框为\s*\[[^\]]*\]", "", cap)
    cap = re.sub(r"（[^）]*边界框[^）]*）", "", cap)
    cap = re.sub(r"（[^）]*(?:商品图|实拍图|穿搭图|背景)[^）]*）", "", cap)
    cap = re.sub(r"（约在画面[^）]*）", "", cap)
    cap = re.sub(r"（[，、；\s]*）", "", cap)
    cap = cap.replace("贴附有", "印有").replace("实拍商品", "商品")
    cap = cap.replace("与底布形成自然的丝印/热转印质感", "印花与面料融合自然")
    cap = cap.replace("与底布形成的丝印/热转印质感", "印花与面料融合自然")
    for src, dst in {"下方右侧": "右下方", "下方左侧": "左下方", "上方右侧": "右上方", "上方左侧": "左上方"}.items():
        cap = cap.replace(f"位于主图案的{src}", f"位于主图案{dst}")
    cap = re.sub(r"\s+【", "【", cap)
    cap = re.sub(r"[，；]?。", "。", cap)
    cap = re.sub(r"；\s*；", "；", cap)
    cap = re.sub(r"\s{2,}", " ", cap).strip()
    return cap


def clean_v2_caption(row: Dict[str, Any]) -> Dict[str, Any]:
    """B/C/D/E: 去掉场景括注、知识扩写句、caption 里的 bbox 与重复句。"""
    cap = clean_free_caption(row.get("caption_zh", ""))
    if cap:
        row["caption_zh"] = cap
    return row


V2_TEXT_FIELDS = [
    "stage_title_zh", "caption_zh", "caption_en", "visual_analysis",
    "knowledge_expansion", "qa_pairs_json", "entities_json",
    "substrate_category_zh", "scene_style_zh",
]


def normalize_shard_files(shards_dir: Path) -> int:
    files = sorted(shards_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no shard jsonl under {shards_dir}")
    changed = 0
    for path in files:
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("task_stage") == "stage_a_geometry":
                    row = rewrite_stage_a_row(row)
                for field in V2_TEXT_FIELDS:
                    if field in row:
                        row[field] = normalize_text(row[field])
                if row.get("task_stage") != "stage_a_geometry":
                    row = clean_v2_caption(row)
                if row.get("substrate_category_zh") in CATEGORY_MAP:
                    row["substrate_category_zh"] = CATEGORY_MAP[row["substrate_category_zh"]]
                if row.get("scene_style_zh") in SCENE_MAP:
                    row["scene_style_zh"] = SCENE_MAP[row["scene_style_zh"]]
                vis = row.get("visual_analysis", "")
                if "图中有 张" in vis:
                    try:
                        refs = row.get("ref_candidates_json") or row.get("ref_candidates") or "[]"
                        n = len(json.loads(refs) if isinstance(refs, str) else refs)
                    except Exception:
                        n = 0
                    row["visual_analysis"] = vis.replace("图中有 张", f"图中有 {n} 张" if n else "图中有若干张")
                rows.append(row)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        changed += len(rows)
        print(f"[normalize] shard {path.name}: {len(rows)} rows")
    return changed


def normalize_parquet(path: Path) -> int:
    if not path.exists():
        print(f"[normalize] skip missing {path}")
        return 0
    table = pq.read_table(path)
    names = table.schema.names
    rows = table.to_pylist()
    for row in rows:
        if "messages" in names:
            msgs = []
            for m in row.get("messages") or []:
                mm = dict(m)
                content = mm.get("content", "")
                stripped = content.strip()
                if stripped.startswith("{") and '"caption_zh"' in stripped:
                    try:
                        obj = json.loads(stripped)
                        if isinstance(obj, dict) and isinstance(obj.get("caption_zh"), str):
                            obj["caption_zh"] = clean_free_caption(obj["caption_zh"])
                            if obj.get("scene_style_zh") in SCENE_MAP:
                                obj["scene_style_zh"] = SCENE_MAP[obj["scene_style_zh"]]
                            content = json.dumps(obj, ensure_ascii=False)
                    except Exception:
                        pass
                mm["content"] = normalize_text(content)
                msgs.append(mm)
            row["messages"] = msgs
        for field in ("caption_zh", "caption_en", "visual_analysis", "knowledge_expansion"):
            if field in row:
                row[field] = normalize_text(row[field])
    out = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(out, path, compression="zstd")
    print(f"[normalize] parquet {path.relative_to(PROJECT_ROOT)}: {len(rows)} rows")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Datawheel dataset language")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-v6", action="store_true")
    parser.add_argument("--extra-dataset", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    if not args.skip_v2:
        normalize_shard_files(V2_SHARDS)
    if not args.skip_v6:
        for part in V6_PARTS:
            normalize_parquet(part)
    for extra in args.extra_dataset:
        normalize_parquet(extra)
    print("[normalize] done; remember to rebuild v2 (--merge) and V6 (pack/assemble)")


if __name__ == "__main__":
    main()
