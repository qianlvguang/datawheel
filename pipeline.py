"""飞轮流水线主入口 (Pipeline Generator)。
自动化批量生成图像、密集标注 JSON 以及多轮问答对数据集。
可直接对接 ms-swift / Megatron-Core 训练格式。
"""
import os
import json
from pathlib import Path
from PIL import Image

from engine.builder import FlagBuilder
from engine.modifiers import apply_cutout_or_hollowing, splice_two_flags
from synthesizer.ground_truth import GroundTruthSynthesizer

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGE_DIR = OUTPUT_DIR / "images"


def run_demo_flywheel_batch(num_samples: int = 6):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    builder = FlagBuilder(width=600, height=400)
    dataset = []

    print(f"[flywheel] 开始批量合成 {num_samples} 个具有代表性的多模态样本...")

    # --- 样本 1: Canton 条纹 + 5 颗环形星 (方位 + 数量 + 颜色) ---
    img1, meta1 = builder.create_base("canton", ["red", "white", "navy_blue"])
    elem1 = builder.add_emblem_or_symbols(img1, element_type="stars", element_color="yellow",
                                          position="canton", count=5)
    meta1.update(elem1)
    path1 = IMAGE_DIR / "sample_01_canton_stars.png"
    img1.save(path1)
    gt1 = GroundTruthSynthesizer.build_structured_gt(meta1)
    cap1 = GroundTruthSynthesizer.build_dense_caption(meta1, gt1)
    qa1 = GroundTruthSynthesizer.build_qa_dataset(meta1, gt1)
    dataset.append({"id": "sample_01", "image": str(path1), "gt": gt1, "caption": cap1, "qa": qa1})

    # --- 样本 2: 纯绿底板 + 新月与星 (伊斯兰几何元素 + 角度 + 颜色) ---
    img2, meta2 = builder.create_base("solid_field", ["dark_green"])
    elem2 = builder.add_emblem_or_symbols(img2, element_type="crescent_and_star", element_color="white",
                                          position="center", count=1)
    meta2.update(elem2)
    path2 = IMAGE_DIR / "sample_02_crescent_star.png"
    img2.save(path2)
    gt2 = GroundTruthSynthesizer.build_structured_gt(meta2)
    cap2 = GroundTruthSynthesizer.build_dense_caption(meta2, gt2)
    qa2 = GroundTruthSynthesizer.build_qa_dataset(meta2, gt2)
    dataset.append({"id": "sample_02", "image": str(path2), "gt": gt2, "caption": cap2, "qa": qa2})

    # --- 样本 3: 北欧十字 + 16 道光芒太阳 (十字方位 + 放射数量) ---
    img3, meta3 = builder.create_base("nordic_cross", ["blue", "yellow"])
    elem3 = builder.add_emblem_or_symbols(img3, element_type="sun", element_color="gold",
                                          position="hoist", count=1)
    meta3.update(elem3)
    path3 = IMAGE_DIR / "sample_03_nordic_sun.png"
    img3.save(path3)
    gt3 = GroundTruthSynthesizer.build_structured_gt(meta3)
    cap3 = GroundTruthSynthesizer.build_dense_caption(meta3, gt3)
    qa3 = GroundTruthSynthesizer.build_qa_dataset(meta3, gt3)
    dataset.append({"id": "sample_03", "image": str(path3), "gt": gt3, "caption": cap3, "qa": qa3})

    # --- 样本 4: 三色横条纹 + 纹章盾徽 + 飘带文字 (徽章识别 + 文字 OCR) ---
    img4, meta4 = builder.create_base("horizontal_triband", ["black", "red", "gold"])
    elem4 = builder.add_emblem_or_symbols(img4, element_type="coat_of_arms", element_color="dark_red",
                                          position="center", count=1, text="UNITAS ET LIBERTAS")
    meta4.update(elem4)
    path4 = IMAGE_DIR / "sample_04_emblem_ocr.png"
    img4.save(path4)
    gt4 = GroundTruthSynthesizer.build_structured_gt(meta4)
    cap4 = GroundTruthSynthesizer.build_dense_caption(meta4, gt4)
    qa4 = GroundTruthSynthesizer.build_qa_dataset(meta4, gt4)
    dataset.append({"id": "sample_04", "image": str(path4), "gt": gt4, "caption": cap4, "qa": qa4})

    # --- 样本 5: 挖空缺损旗帜 (完整性破坏识别 / 空间感知) ---
    img5_base, meta5_base = builder.create_base("horizontal_biband", ["red", "white"])
    elem5 = builder.add_emblem_or_symbols(img5_base, element_type="stars", element_color="blue",
                                          position="center", count=1)
    meta5_base.update(elem5)
    img5, cutout_meta = apply_cutout_or_hollowing(img5_base, cutout_type="circular_cutout")
    meta5_base.update(cutout_meta)
    path5 = IMAGE_DIR / "sample_05_cutout_hollow.png"
    img5.save(path5)
    gt5 = GroundTruthSynthesizer.build_structured_gt(meta5_base)
    cap5 = GroundTruthSynthesizer.build_dense_caption(meta5_base, gt5)
    qa5 = GroundTruthSynthesizer.build_qa_dataset(meta5_base, gt5)
    dataset.append({"id": "sample_05", "image": str(path5), "gt": gt5, "caption": cap5, "qa": qa5})

    # --- 样本 6: 左右两面旗帜拼接 (拼接来源溯源 / 复合识别) ---
    img_left, meta_left = builder.create_base("vertical_triband", ["green", "white", "red"])
    img_right, meta_right = builder.create_base("diagonal_bend", ["sky_blue", "crimson", "yellow"])
    img6, splice_meta = splice_two_flags(img_left, meta_left, img_right, meta_right, split_mode="vertical")
    path6 = IMAGE_DIR / "sample_06_spliced_hybrid.png"
    img6.save(path6)
    gt6 = GroundTruthSynthesizer.build_structured_gt(splice_meta)
    cap6 = GroundTruthSynthesizer.build_dense_caption(splice_meta, gt6)
    qa6 = GroundTruthSynthesizer.build_qa_dataset(splice_meta, gt6)
    dataset.append({"id": "sample_06", "image": str(path6), "gt": gt6, "caption": cap6, "qa": qa6})

    # ==================== 4 大高阶认知任务 (数星星 / 找不同 / 加减法 / 多增少补) ====================
    from engine.cognitive_tasks import CognitiveTaskEngine

    # --- 样本 7: 数星星与离散分布 (精准计数 + 0~1000 相对坐标定位) ---
    img7, meta7 = CognitiveTaskEngine.generate_star_counting_task(width=600, height=400, star_count=8)
    path7 = IMAGE_DIR / "sample_07_star_counting.png"
    img7.save(path7)
    gt7 = GroundTruthSynthesizer.build_structured_gt(meta7)
    cap7 = GroundTruthSynthesizer.build_dense_caption(meta7, gt7)
    qa7 = GroundTruthSynthesizer.build_qa_dataset(meta7, gt7)
    dataset.append({"id": "sample_07", "image": str(path7), "gt": gt7, "caption": cap7, "qa": qa7})

    # --- 样本 8: 对比图与找不同 (左右双联画对比，标出 0~1000 变动区域) ---
    img8, meta8 = CognitiveTaskEngine.generate_spot_the_difference_pair(width=400, height=400, diff_type="missing_element")
    path8 = IMAGE_DIR / "sample_08_spot_the_difference.png"
    img8.save(path8)
    gt8 = GroundTruthSynthesizer.build_structured_gt(meta8)
    cap8 = GroundTruthSynthesizer.build_dense_caption(meta8, gt8)
    qa8 = GroundTruthSynthesizer.build_qa_dataset(meta8, gt8)
    dataset.append({"id": "sample_08", "image": str(path8), "gt": gt8, "caption": cap8, "qa": qa8})

    # --- 样本 9: 视觉加减法对比 (左图3颗星 + 右图5颗星 = 合计8颗星，差值2颗) ---
    img9, meta9 = CognitiveTaskEngine.generate_visual_arithmetic_pair(width=400, height=400, count_a=3, count_b=5)
    path9 = IMAGE_DIR / "sample_09_visual_arithmetic.png"
    img9.save(path9)
    gt9 = GroundTruthSynthesizer.build_structured_gt(meta9)
    cap9 = GroundTruthSynthesizer.build_dense_caption(meta9, gt9)
    qa9 = GroundTruthSynthesizer.build_qa_dataset(meta9, gt9)
    dataset.append({"id": "sample_09", "image": str(path9), "gt": gt9, "caption": cap9, "qa": qa9})

    # --- 样本 10: 多增少补 (目标标准5颗星，实际渲染3颗星，判定缺少2颗需补充) ---
    img10, meta10 = CognitiveTaskEngine.generate_surplus_deficit_task(width=600, height=400, target_standard_count=5, actual_count=3)
    path10 = IMAGE_DIR / "sample_10_surplus_deficit.png"
    img10.save(path10)
    gt10 = GroundTruthSynthesizer.build_structured_gt(meta10)
    cap10 = GroundTruthSynthesizer.build_dense_caption(meta10, gt10)
    qa10 = GroundTruthSynthesizer.build_qa_dataset(meta10, gt10)
    dataset.append({"id": "sample_10", "image": str(path10), "gt": gt10, "caption": cap10, "qa": qa10})

    # 导出 JSONL 数据集
    jsonl_path = OUTPUT_DIR / "datawheel_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[datawheel] 数据飞轮全批次构建完成！")
    print(f"  - 生成图片目录: {IMAGE_DIR}")
    print(f"  - 数据集落盘路径: {jsonl_path} (共 {len(dataset)} 条高纯度全能力样本)")
    return jsonl_path


if __name__ == "__main__":
    run_demo_flywheel_batch()
