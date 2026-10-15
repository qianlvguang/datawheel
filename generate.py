"""Datawheel 数据飞轮总控生成器 (Unified Dataset Generator CLI)。
聚合全域 4 大生成维度，支持单维度抽取与全流水线一键构建：
1. gallery: 官方 SVG 权威二创与 8 大认知任务画廊 (24 样本)
2. contrastive: 七态物理实物对比矩阵与跨图参考微调数据集 (21 样本)
3. density: 原创模块化积木实体与 5 梯度空间与物理做旧浓度阶梯 (5 样本)
4. demo: 基础几何旗帜与认知任务快速冒烟验证 (10 样本)

使用命令：
  python generate.py                  # 一键生成全部 4 大维度并汇总 master_dataset.jsonl
  python generate.py gallery          # 仅生成官方 SVG 8 大认知画廊
  python generate.py contrastive      # 仅生成七态物理实物对比矩阵
  python generate.py density          # 仅生成 5 梯度浓度阶梯实验
  python generate.py demo             # 仅生成经典 Demo 冒烟验证
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from engine.coords import draw_debug_bboxes
from engine.svg_remix import OfficialSVGRemixEngine
from engine.reference_bank import ReferenceBank
from engine.contrastive_generator import ContrastiveGenerator
from engine.modular_entity import ModularEntityBuilder
from engine.density_engine import ContentAndDensityEngine
from engine.builder import FlagBuilder
from engine.modifiers import apply_cutout_or_hollowing, splice_two_flags
from engine.cognitive_tasks import CognitiveTaskEngine
from synthesizer.ground_truth import GroundTruthSynthesizer
from synthesizer.reference_qa import ReferenceQASynthesizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


def _save_jsonl(records: List[Dict[str, Any]], filepath: Path) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return filepath


def _save_debug(image: Image.Image, boxes: Any, out_path: Path, label: str = ""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dbg = draw_debug_bboxes(image, boxes, label=label)
    fmt = "JPEG" if out_path.suffix.lower() in [".jpg", ".jpeg"] else "PNG"
    dbg.save(out_path, format=fmt, quality=90)


# =============================================================================
# 1. 官方 SVG 8 大认知任务画廊生成器 (Gallery Generator)
# =============================================================================
def generate_gallery(output_dir: Path = None, debug: bool = True) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "gallery"
    img_dir = out_dir / "images"
    dbg_dir = out_dir / "visual_debug"
    jsonl_path = out_dir / "gallery_dataset.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    print("\n[gallery] 正在基于权威官方标准 SVG 构建 8 大类 24 个代表性样本...")

    tasks = [
        ("数星星", "cat1_01", "cat1_star_counting_01_china.png", lambda: OfficialSVGRemixEngine.task_star_counting("cn")),
        ("数星星", "cat1_02", "cat1_star_counting_02_eu.png", lambda: OfficialSVGRemixEngine.task_star_counting("eu")),
        ("数星星", "cat1_03", "cat1_star_counting_03_somalia.png", lambda: OfficialSVGRemixEngine.task_star_counting("so")),
        ("找不同", "cat2_01", "cat2_spot_diff_01_china_missing_star.png", lambda: OfficialSVGRemixEngine.task_spot_the_difference("cn", "missing_star")),
        ("找不同", "cat2_02", "cat2_spot_diff_02_spain_missing_emblem.png", lambda: OfficialSVGRemixEngine.task_spot_the_difference("es", "missing_emblem")),
        ("找不同", "cat2_03", "cat2_spot_diff_03_germany_stripe_color.png", lambda: OfficialSVGRemixEngine.task_spot_the_difference("de", "color_change")),
        ("视觉加减法", "cat3_01", "cat3_arithmetic_01_china_plus_somalia.png", lambda: OfficialSVGRemixEngine.task_visual_arithmetic("cn", "so", 5, 1)),
        ("视觉加减法", "cat3_02", "cat3_arithmetic_02_eu_plus_vietnam.png", lambda: OfficialSVGRemixEngine.task_visual_arithmetic("eu", "vn", 12, 1)),
        ("视觉加减法", "cat3_03", "cat3_arithmetic_03_china_plus_vietnam.png", lambda: OfficialSVGRemixEngine.task_visual_arithmetic("cn", "vn", 5, 1)),
        ("多增少补", "cat4_01", "cat4_balance_01_china_deficit.png", lambda: OfficialSVGRemixEngine.task_surplus_deficit("cn", 5, 3)),
        ("多增少补", "cat4_02", "cat4_balance_02_vietnam_surplus.png", lambda: OfficialSVGRemixEngine.task_surplus_deficit("vn", 1, 2)),
        ("多增少补", "cat4_03", "cat4_balance_03_somalia_balanced.png", lambda: OfficialSVGRemixEngine.task_surplus_deficit("so", 1, 1)),
        ("图元与方位", "cat5_01", "cat5_element_01_switzerland_cross.png", lambda: OfficialSVGRemixEngine.task_real_standard_flag("ch")),
        ("图元与方位", "cat5_02", "cat5_element_02_sweden_nordic_cross.png", lambda: OfficialSVGRemixEngine.task_real_standard_flag("se")),
        ("图元与方位", "cat5_03", "cat5_element_03_somalia_star.png", lambda: OfficialSVGRemixEngine.task_real_standard_flag("so")),
        ("国徽与文字", "cat6_01", "cat6_emblem_01_spain_plvs_vltra.png", lambda: OfficialSVGRemixEngine.task_official_emblem_ocr("es")),
        ("国徽与文字", "cat6_02", "cat6_emblem_02_brazil_ordem_progresso.png", lambda: OfficialSVGRemixEngine.task_official_emblem_ocr("br")),
        ("国徽与文字", "cat6_03", "cat6_emblem_03_mexico_eagle_emblem.png", lambda: OfficialSVGRemixEngine.task_official_emblem_ocr("mx")),
        ("挖空与完整性", "cat7_01", "cat7_cutout_01_france_circle_hole.png", lambda: OfficialSVGRemixEngine.task_real_flag_cutout("fr", "circular_cutout")),
        ("挖空与完整性", "cat7_02", "cat7_cutout_02_ukraine_rect_hole.png", lambda: OfficialSVGRemixEngine.task_real_flag_cutout("ua", "rectangular_cutout")),
        ("挖空与完整性", "cat7_03", "cat7_cutout_03_sweden_corner_tear.png", lambda: OfficialSVGRemixEngine.task_real_flag_cutout("se", "missing_corner")),
        ("拼接来源", "cat8_01", "cat8_spliced_01_france_germany_vert.png", lambda: OfficialSVGRemixEngine.task_real_flag_splicing("fr", "de", "vertical")),
        ("拼接来源", "cat8_02", "cat8_spliced_02_poland_ukraine_horiz.png", lambda: OfficialSVGRemixEngine.task_real_flag_splicing("pl", "ua", "horizontal")),
        ("拼接来源", "cat8_03", "cat8_spliced_03_sweden_switzerland_hybrid.png", lambda: OfficialSVGRemixEngine.task_real_flag_splicing("se", "ch", "vertical")),
    ]

    records = []
    for cat, sid, fname, fn in tasks:
        img, meta = fn()
        p = img_dir / fname
        img.save(p)

        gt = GroundTruthSynthesizer.build_structured_gt(meta)
        cap = GroundTruthSynthesizer.build_dense_caption(meta, gt)
        qa = GroundTruthSynthesizer.build_qa_dataset(meta, gt)

        boxes = meta.get("all_bboxes_1000") or (
            [meta["target_differing_box_right_1000"]] if meta.get("target_differing_box_right_1000") else meta.get("bboxes", [])
        )
        if debug and boxes:
            _save_debug(img, boxes, dbg_dir / f"debug_{fname}", label=cat)

        records.append({"id": sid, "category": cat, "image": str(p), "gt": gt, "caption": cap, "qa": qa})

    _save_jsonl(records, jsonl_path)
    print(f"[gallery] ✅ 画廊构建完成: 共 {len(records)} 样本落盘至 {jsonl_path}")
    return {"name": "gallery", "samples_count": len(records), "images_dir": img_dir, "dataset_path": jsonl_path, "records": records}


# =============================================================================
# 2. 七态物理对比矩阵生成器 (Contrastive Generator)
# =============================================================================
def generate_contrastive(output_dir: Path = None, debug: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "contrastive"
    img_dir = out_dir / "images"
    ref_dir = out_dir / "references"
    dbg_dir = out_dir / "visual_debug"
    jsonl_path = out_dir / "dataset_7_state_sft.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    generator = ContrastiveGenerator(width=640, height=480)
    records = []
    test_cases = [
        {"ref_id": "cn", "substrate": "tshirt_fabric", "seed": seed or 101},
        {"ref_id": "fr", "substrate": "ceramic_mug", "seed": seed or 202},
        {"ref_id": "eu", "substrate": "kraft_paper", "seed": seed or 303}
    ]

    print("\n[contrastive] 正在生成 7 态全矩阵多模态微调数据集...")
    for case in test_cases:
        rid, substrate, cseed = case["ref_id"], case["substrate"], case["seed"]
        meta = ReferenceBank.get_meta(rid)
        print(f"  ▶ [参考对象] {meta['name_zh']} ({rid.upper()}) | 底板材质: {substrate}")

        ref_card = ReferenceBank.render_reference_card(rid, card_size=(400, 300))
        ref_card_path = ref_dir / f"ref_{rid}.png"
        ref_card.save(ref_card_path)

        matrix = generator.generate_7_state_matrix(rid, substrate_type=substrate, base_seed=cseed)
        for sample in matrix:
            sk = sample["state_key"].lower()
            fname = f"{rid}_{sk}.jpg"
            ipath = img_dir / fname
            sample["image"].save(ipath, format="JPEG", quality=90)

            if debug and sample.get("bbox_1000"):
                _save_debug(sample["image"], sample["bbox_1000"], dbg_dir / f"debug_{fname}", label=sample["match_status"])

            qa_rec = ReferenceQASynthesizer.build_in_context_qa(sample, str(ref_card_path), str(ipath))
            records.append(qa_rec)

    _save_jsonl(records, jsonl_path)
    print(f"[contrastive] ✅ 七态对比矩阵完成: 共 {len(records)} 样本落盘至 {jsonl_path}")
    return {"name": "contrastive", "samples_count": len(records), "images_dir": img_dir, "dataset_path": jsonl_path, "records": records}


# =============================================================================
# 3. 内容与浓度阶梯实验生成器 (Density Generator)
# =============================================================================
def generate_density(output_dir: Path = None, debug: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "density"
    img_dir = out_dir / "images"
    ent_dir = out_dir / "entities"
    dbg_dir = out_dir / "visual_debug"
    jsonl_path = out_dir / "dataset_density_sft.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    engine = ContentAndDensityEngine(width=640, height=480)
    target, tmeta = ModularEntityBuilder.build_entity("blob", 3, "horns", "star", "cyber_cyan")
    target.save(ent_dir / "target_cyan_3eyes_horns.png")

    d1, _ = ModularEntityBuilder.build_entity("blob", 1, "none", "star", "cyber_cyan")
    d2, _ = ModularEntityBuilder.build_entity("blob", 2, "crown", "star", "cyber_cyan")
    d3, _ = ModularEntityBuilder.build_entity("blob", 3, "antenna", "star", "coral_pink")
    d1.save(ent_dir / "distractor_cyan_1eye_nohorns.png")
    d2.save(ent_dir / "distractor_cyan_2eyes_crown.png")
    d3.save(ent_dir / "distractor_pink_3eyes_antenna.png")

    experiments = [
        ("exp1_sparse_chest", "极简低空间浓度 (胸标 8%~12% 占比)", "纯棉卫衣浅燕麦底",
         "在商品图上定位左胸口的原创微型印花，并确认其眼数与饰物特征。",
         lambda: engine.render_sparse_chest_motif(target, color_key="heather_grey", seed=seed or 201)),
        ("exp2_medium_compound", "标准中空间浓度 (中心大幅印花 30% 占比)", "纯棉 T 恤大幅印花",
         "分析中心复合印花的主体，判断是否与目标三眼怪兽一致。",
         lambda: engine.render_medium_compound(target, color_key="white", seed=seed or 202)),
        ("exp3_dense_monogram", "满版极高空间浓度 (老花重复阵列 >70% 占比)", "环保棉帆布托特包全包身平铺",
         "分析整幅满版重复平铺的老花图案，识别构成该图案的核心基元实体。",
         lambda: engine.render_dense_monogram_allover(target, color_key="canvas_beige", tile_size=65, seed=seed or 203)),
        ("exp4_washed_vintage_fade", "稀释做旧物理浓度 (50% 透底水洗 + 龟裂磨损)", "曜石黑连帽卫衣胸前水洗龟裂",
         "印花出现严重掉色透底与表面龟裂，请判定残存关键轮廓是否仍符合目标实体。",
         lambda: engine.render_washed_vintage_fade(target, color_key="black", opacity=0.50, seed=seed or 204)),
        ("exp5_cluttered_multi_distractors", "多干扰项密集混淆浓度 (4 槽位多目标排他定位)", "藏青色连帽卫衣前襟 4 槽位",
         "商品表面印有多只相似怪兽，请排除单眼、双眼及粉色怪兽，唯一定位【三眼、带双弯角】的青色怪兽。",
         lambda: engine.render_cluttered_multi_distractors(target, [d1, d2, d3], color_key="navy_blue", seed=seed or 205)),
    ]

    print("\n[density] 开始执行 5 梯度空间与做旧物理浓度渲染...")
    records = []
    for did, dname, substrate, prompt, fn in experiments:
        img, annotations, meta = fn()
        fname = f"{did}.jpg"
        ipath = img_dir / fname
        img.save(ipath, format="JPEG", quality=90)

        if debug and annotations:
            _save_debug(img, annotations, dbg_dir / f"debug_{fname}")

        targets = [a for a in annotations if a.get("is_target", True)]
        tbox = (targets[0] if targets else annotations[0]).get("bbox_1000")

        assistant_answer = {
            "experiment": did,
            "density_level": dname,
            "target_detected": True,
            "primary_target_bbox_1000": tbox,
            "discriminative_verification": {
                "eye_count": tmeta["eye_count"],
                "accessory": tmeta["accessory_desc"],
                "body_style": tmeta["body_desc"],
                "chest_symbol": tmeta["chest_desc"]
            },
            "total_objects_in_scene": len(annotations),
            "scene_description": f"在{substrate}上成功完成{dname}维度的可控定位。"
        }
        rec = {
            "id": did,
            "density_type": did,
            "image": str(ipath),
            "conversations": [
                {"role": "user", "content": f"<|image|>\n{prompt}\n请给出目标实体的 0~1000 相对定位坐标 [ymin, xmin, ymax, xmax] 并列举判定依据。"},
                {"role": "assistant", "content": f"```json\n{json.dumps(assistant_answer, ensure_ascii=False, indent=2)}\n```\n\n**定位结论**：已在目标区域 `{tbox}` 检测到符合【{tmeta['discriminative_summary']}】的实体。"}
            ],
            "annotations": annotations
        }
        records.append(rec)
        print(f"  🎯 [{did}] 目标坐标: {tbox} | 场景目标数: {len(annotations)}")

    _save_jsonl(records, jsonl_path)
    print(f"[density] ✅ 浓度阶梯完成: 共 {len(records)} 样本落盘至 {jsonl_path}")
    return {"name": "density", "samples_count": len(records), "images_dir": img_dir, "dataset_path": jsonl_path, "records": records}


# =============================================================================
# 4. 经典几何与认知 Demo 生成器 (Demo Generator)
# =============================================================================
def generate_demo(output_dir: Path = None, debug: bool = True) -> Dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "demo"
    img_dir = out_dir / "images"
    jsonl_path = out_dir / "demo_dataset.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)

    builder = FlagBuilder(width=600, height=400)
    dataset = []
    print("\n[demo] 开始快速构建 10 个经典几何与认知多模态样本...")

    def _add_sample(sid, fname, img, meta):
        p = img_dir / fname
        img.save(p)
        gt = GroundTruthSynthesizer.build_structured_gt(meta)
        dataset.append({
            "id": sid, "image": str(p), "gt": gt,
            "caption": GroundTruthSynthesizer.build_dense_caption(meta, gt),
            "qa": GroundTruthSynthesizer.build_qa_dataset(meta, gt)
        })

    # 1: Canton 条纹 + 5 星
    img1, m1 = builder.create_base("canton", ["red", "white", "navy_blue"])
    m1.update(builder.add_emblem_or_symbols(img1, "stars", "yellow", "canton", 5))
    _add_sample("sample_01", "sample_01_canton_stars.png", img1, m1)

    # 2: 纯绿底板 + 新月与星
    img2, m2 = builder.create_base("solid_field", ["dark_green"])
    m2.update(builder.add_emblem_or_symbols(img2, "crescent_and_star", "white", "center", 1))
    _add_sample("sample_02", "sample_02_crescent_star.png", img2, m2)

    # 3: 北欧十字 + 太阳
    img3, m3 = builder.create_base("nordic_cross", ["blue", "yellow"])
    m3.update(builder.add_emblem_or_symbols(img3, "sun", "gold", "hoist", 1))
    _add_sample("sample_03", "sample_03_nordic_sun.png", img3, m3)

    # 4: 三色横条纹 + 盾徽 + OCR
    img4, m4 = builder.create_base("horizontal_triband", ["black", "red", "gold"])
    m4.update(builder.add_emblem_or_symbols(img4, "coat_of_arms", "dark_red", "center", 1, text="UNITAS ET LIBERTAS"))
    _add_sample("sample_04", "sample_04_emblem_ocr.png", img4, m4)

    # 5: 挖空缺损旗帜
    img5_b, m5 = builder.create_base("horizontal_biband", ["red", "white"])
    m5.update(builder.add_emblem_or_symbols(img5_b, "stars", "blue", "center", 1))
    img5, c_meta = apply_cutout_or_hollowing(img5_b, "circular_cutout")
    m5.update(c_meta)
    _add_sample("sample_05", "sample_05_cutout_hollow.png", img5, m5)

    # 6: 左右双旗拼接
    img_l, ml = builder.create_base("vertical_triband", ["green", "white", "red"])
    img_r, mr = builder.create_base("diagonal_bend", ["sky_blue", "crimson", "yellow"])
    img6, smeta = splice_two_flags(img_l, ml, img_r, mr, split_mode="vertical")
    _add_sample("sample_06", "sample_06_spliced_hybrid.png", img6, smeta)

    # 7~10: 高阶认知任务
    img7, m7 = CognitiveTaskEngine.generate_star_counting_task(600, 400, 8)
    _add_sample("sample_07", "sample_07_star_counting.png", img7, m7)
    img8, m8 = CognitiveTaskEngine.generate_spot_the_difference_pair(400, 400, "missing_element")
    _add_sample("sample_08", "sample_08_spot_the_difference.png", img8, m8)
    img9, m9 = CognitiveTaskEngine.generate_visual_arithmetic_pair(400, 400, 3, 5)
    _add_sample("sample_09", "sample_09_visual_arithmetic.png", img9, m9)
    img10, m10 = CognitiveTaskEngine.generate_surplus_deficit_task(600, 400, 5, 3)
    _add_sample("sample_10", "sample_10_surplus_deficit.png", img10, m10)

    _save_jsonl(dataset, jsonl_path)
    print(f"[demo] ✅ 经典 Demo 完成: 共 {len(dataset)} 样本落盘至 {jsonl_path}")
    return {"name": "demo", "samples_count": len(dataset), "images_dir": img_dir, "dataset_path": jsonl_path, "records": dataset}


# =============================================================================
# 全流水线与 CLI
# =============================================================================
GENERATORS_MAP = {
    "gallery": generate_gallery,
    "contrastive": generate_contrastive,
    "density": generate_density,
    "demo": generate_demo,
}

ALIASES = {
    "flywheel": "contrastive",
    "flywheel_v2": "contrastive",
    "v2": "contrastive",
    "experiments": "density",
    "classic": "demo",
    "v1": "demo",
}


def run_all(output_dir: Path = None, debug: bool = True, seed: Optional[int] = None,
            merge_master: bool = True) -> Dict[str, Any]:
    base_out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    results = {}
    master_records = []

    print("================================================================================")
    print(" 🚀 Datawheel 工业级全套多模态数据飞轮流水线开始构建")
    print("================================================================================")

    for mode, fn in GENERATORS_MAP.items():
        sub_dir = base_out / mode
        res = fn(output_dir=sub_dir, debug=debug, **({"seed": seed} if "seed" in fn.__code__.co_varnames else {}))
        results[mode] = res
        master_records.extend(res["records"])

    master_path = None
    if merge_master and master_records:
        master_path = base_out / "datawheel_master_dataset.jsonl"
        _save_jsonl(master_records, master_path)
        print(f"\n[master] 🌐 全域统一总数据集落盘: {master_path} (共包含 {len(master_records)} 条高质量样本)")

    return {"results": results, "total_samples": len(master_records), "master_dataset_path": master_path}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Datawheel 程序化多模态数据飞轮统一生成流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持模式:
  all          一键执行全部四大维度并输出 master_dataset.jsonl (默认)
  gallery      官方 SVG 8 大认知任务画廊 (24 样本)
  contrastive  七态物理商品对比矩阵与参考检索微调集 (21 样本)
  density      原创积木实体与 5 梯度空间与做旧浓度阶梯实验 (5 样本)
  demo         经典几何与认知任务冒烟验证 (10 样本)
        """
    )
    parser.add_argument("mode", nargs="?", default=None, help="生成模式: all, gallery, contrastive, density, demo")
    parser.add_argument("-m", "--mode", dest="opt_mode", default=None, help="生成模式: all, gallery, contrastive, density, demo (选项参数形式)")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出根目录路径 (默认: output/)")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True, help="是否绘制带 0~1000 相对 BBox 的质检图")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子")
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True, help="全量运行时是否导出 master_dataset.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_mode = args.opt_mode or args.mode or "all"
    mode = ALIASES.get(selected_mode.lower(), selected_mode.lower())

    if mode == "all":
        res = run_all(output_dir=args.output_dir, debug=args.debug, seed=args.seed, merge_master=args.merge)
        print("\n================================================================================")
        print(" 🎉 全域多模态飞轮数据流水线执行完毕！")
        print(f" 📦 总样本条数: {res['total_samples']}")
        if res["master_dataset_path"]:
            print(f" 📄 统一总微调数据集: {res['master_dataset_path']}")
        print("================================================================================")
    elif mode in GENERATORS_MAP:
        fn = GENERATORS_MAP[mode]
        sub_dir = args.output_dir / mode
        kwargs = {"output_dir": sub_dir, "debug": args.debug}
        if "seed" in fn.__code__.co_varnames:
            kwargs["seed"] = args.seed
        res = fn(**kwargs)
        print("\n================================================================================")
        print(f" ✨ [{res['name']}] 生成完毕！样本数: {res['samples_count']} | 数据集: {res['dataset_path']}")
        print("================================================================================")
    else:
        print(f"❌ 未知模式: '{args.mode}'。可选模式: {list(GENERATORS_MAP.keys())} 或 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
