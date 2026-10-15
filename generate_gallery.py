"""画廊生成脚本 (Gallery Batch Generator - Official SVG Edition)。
全部基于真实权威标准官方 SVG (ISO 3166-1 国旗与官方国徽矢量图) 重新渲染生成！
旧数据作废，重新 roll 出 8 个大类、每类 3 个样本 (共 24 个真实官方二创样本)。
全部 BBox 统一遵循以左上角为原点、0~1000 的相对坐标系 [ymin, xmin, ymax, xmax]。
"""
import shutil
import json
from pathlib import Path

from engine.svg_remix import OfficialSVGRemixEngine
from synthesizer.ground_truth import GroundTruthSynthesizer

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "gallery"
DATASET_PATH = OUTPUT_DIR.parent / "gallery_dataset.jsonl"


def generate_official_gallery():
    # 彻底作废并清理旧数据
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    print("[official_gallery] 正在基于权威官方标准 SVG 重新 roll 24 个真实样本...")

    # =========================================================================
    # 类别 1: 真实国旗数星星 (Star Counting) x 3
    # =========================================================================
    # 1.1: 中国国旗 (5 颗星)
    img, meta = OfficialSVGRemixEngine.task_star_counting("cn")
    p = OUTPUT_DIR / "cat1_star_counting_01_china.png"
    img.save(p)
    samples.append({"category": "数星星", "id": "cat1_01", "img": str(p), "meta": meta})

    # 1.2: 欧盟旗帜 (12 颗星环)
    img, meta = OfficialSVGRemixEngine.task_star_counting("eu")
    p = OUTPUT_DIR / "cat1_star_counting_02_eu.png"
    img.save(p)
    samples.append({"category": "数星星", "id": "cat1_02", "img": str(p), "meta": meta})

    # 1.3: 索马里国旗 (1 颗中央独立星)
    img, meta = OfficialSVGRemixEngine.task_star_counting("so")
    p = OUTPUT_DIR / "cat1_star_counting_03_somalia.png"
    img.save(p)
    samples.append({"category": "数星星", "id": "cat1_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 2: 真实国旗对比图找不同 (Spot the Difference) x 3
    # =========================================================================
    # 2.1: 中国国旗 (右图抹除最下方一颗小星)
    img, meta = OfficialSVGRemixEngine.task_spot_the_difference("cn", "missing_star")
    p = OUTPUT_DIR / "cat2_spot_diff_01_china_missing_star.png"
    img.save(p)
    samples.append({"category": "找不同", "id": "cat2_01", "img": str(p), "meta": meta})

    # 2.2: 西班牙国旗 (右图抹除官方皇家国徽与大力神柱)
    img, meta = OfficialSVGRemixEngine.task_spot_the_difference("es", "missing_emblem")
    p = OUTPUT_DIR / "cat2_spot_diff_02_spain_missing_emblem.png"
    img.save(p)
    samples.append({"category": "找不同", "id": "cat2_02", "img": str(p), "meta": meta})

    # 2.3: 德国国旗 (右图将底部金色条纹篡改为天蓝色)
    img, meta = OfficialSVGRemixEngine.task_spot_the_difference("de", "color_change")
    p = OUTPUT_DIR / "cat2_spot_diff_03_germany_stripe_color.png"
    img.save(p)
    samples.append({"category": "找不同", "id": "cat2_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 3: 真实国旗视觉加减法 (Visual Arithmetic) x 3
    # =========================================================================
    # 3.1: 中国(5星) + 索马里(1星) = 6星 (差值4)
    img, meta = OfficialSVGRemixEngine.task_visual_arithmetic("cn", "so", 5, 1)
    p = OUTPUT_DIR / "cat3_arithmetic_01_china_plus_somalia.png"
    img.save(p)
    samples.append({"category": "视觉加减法", "id": "cat3_01", "img": str(p), "meta": meta})

    # 3.2: 欧盟(12星) + 越南(1星) = 13星 (差值11)
    img, meta = OfficialSVGRemixEngine.task_visual_arithmetic("eu", "vn", 12, 1)
    p = OUTPUT_DIR / "cat3_arithmetic_02_eu_plus_vietnam.png"
    img.save(p)
    samples.append({"category": "视觉加减法", "id": "cat3_02", "img": str(p), "meta": meta})

    # 3.3: 中国(5星) + 越南(1星) = 6星 (差值4)
    img, meta = OfficialSVGRemixEngine.task_visual_arithmetic("cn", "vn", 5, 1)
    p = OUTPUT_DIR / "cat3_arithmetic_03_china_plus_vietnam.png"
    img.save(p)
    samples.append({"category": "视觉加减法", "id": "cat3_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 4: 真实国旗多增少补与规则比对 (Surplus & Deficit) x 3
    # =========================================================================
    # 4.1: 中国国旗标准5星，当前只有3星 (缺少2颗，需补2颗)
    img, meta = OfficialSVGRemixEngine.task_surplus_deficit("cn", target_count=5, actual_count=3)
    p = OUTPUT_DIR / "cat4_balance_01_china_deficit.png"
    img.save(p)
    samples.append({"category": "多增少补", "id": "cat4_01", "img": str(p), "meta": meta})

    # 4.2: 越南国旗标准1星，画面多画了1颗变2星 (多增1颗，需删除1颗)
    img, meta = OfficialSVGRemixEngine.task_surplus_deficit("vn", target_count=1, actual_count=2)
    p = OUTPUT_DIR / "cat4_balance_02_vietnam_surplus.png"
    img.save(p)
    samples.append({"category": "多增少补", "id": "cat4_02", "img": str(p), "meta": meta})

    # 4.3: 索马里国旗标准1星，当前恰好1星 (刚好吻合)
    img, meta = OfficialSVGRemixEngine.task_surplus_deficit("so", target_count=1, actual_count=1)
    p = OUTPUT_DIR / "cat4_balance_03_somalia_balanced.png"
    img.save(p)
    samples.append({"category": "多增少补", "id": "cat4_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 5: 真实经典国旗图元、色彩与结构 (Real Flag Base Recognition) x 3
    # =========================================================================
    # 5.1: 瑞士国旗 (鲜红底板 + 正中央纯白正十字)
    img, meta = OfficialSVGRemixEngine.task_real_standard_flag("ch")
    p = OUTPUT_DIR / "cat5_element_01_switzerland_cross.png"
    img.save(p)
    samples.append({"category": "图元与方位", "id": "cat5_01", "img": str(p), "meta": meta})

    # 5.2: 瑞典国旗 (天蓝底板 + 黄色北欧十字偏置)
    img, meta = OfficialSVGRemixEngine.task_real_standard_flag("se")
    p = OUTPUT_DIR / "cat5_element_02_sweden_nordic_cross.png"
    img.save(p)
    samples.append({"category": "图元与方位", "id": "cat5_02", "img": str(p), "meta": meta})

    # 5.3: 索马里国旗 (天蓝底板 + 白色中央独立星)
    img, meta = OfficialSVGRemixEngine.task_real_standard_flag("so")
    p = OUTPUT_DIR / "cat5_element_03_somalia_star.png"
    img.save(p)
    samples.append({"category": "图元与方位", "id": "cat5_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 6: 官方国徽纹章与真实铭文 OCR (Official Emblem & Text OCR) x 3
    # =========================================================================
    # 6.1: 西班牙官方国旗 (官方国徽 + 大力神柱飘带铭文 "PLVS VLTRA")
    img, meta = OfficialSVGRemixEngine.task_official_emblem_ocr("es")
    p = OUTPUT_DIR / "cat6_emblem_01_spain_plvs_vltra.png"
    img.save(p)
    samples.append({"category": "国徽与文字", "id": "cat6_01", "img": str(p), "meta": meta})

    # 6.2: 巴西官方国旗 (天球仪 + 拱形白色飘带铭文 "ORDEM E PROGRESSO")
    img, meta = OfficialSVGRemixEngine.task_official_emblem_ocr("br")
    p = OUTPUT_DIR / "cat6_emblem_02_brazil_ordem_progresso.png"
    img.save(p)
    samples.append({"category": "国徽与文字", "id": "cat6_02", "img": str(p), "meta": meta})

    # 6.3: 墨西哥官方国旗 (绿色、白色、红色垂直三色 + 中央金雕吃蛇国徽)
    img, meta = OfficialSVGRemixEngine.task_official_emblem_ocr("mx")
    p = OUTPUT_DIR / "cat6_emblem_03_mexico_eagle_emblem.png"
    img.save(p)
    samples.append({"category": "国徽与文字", "id": "cat6_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 7: 真实国旗空间挖空与完整性破坏 (Real Flag Cutouts & Completeness) x 3
    # =========================================================================
    # 7.1: 法国官方国旗 (中央圆形孔洞镂空)
    img, meta = OfficialSVGRemixEngine.task_real_flag_cutout("fr", "circular_cutout")
    p = OUTPUT_DIR / "cat7_cutout_01_france_circle_hole.png"
    img.save(p)
    samples.append({"category": "挖空与完整性", "id": "cat7_01", "img": str(p), "meta": meta})

    # 7.2: 乌克兰官方国旗 (矩形挖空撕裂)
    img, meta = OfficialSVGRemixEngine.task_real_flag_cutout("ua", "rectangular_cutout")
    p = OUTPUT_DIR / "cat7_cutout_02_ukraine_rect_hole.png"
    img.save(p)
    samples.append({"category": "挖空与完整性", "id": "cat7_02", "img": str(p), "meta": meta})

    # 7.3: 瑞典官方国旗 (右上角斜向残缺缺角)
    img, meta = OfficialSVGRemixEngine.task_real_flag_cutout("se", "missing_corner")
    p = OUTPUT_DIR / "cat7_cutout_03_sweden_corner_tear.png"
    img.save(p)
    samples.append({"category": "挖空与完整性", "id": "cat7_03", "img": str(p), "meta": meta})

    # =========================================================================
    # 类别 8: 真实跨国旗拼接来源识别 (Real Flag Splicing & Hybrid) x 3
    # =========================================================================
    # 8.1: 左半边法国(竖条纹) + 右半边德国(横条纹) 左右垂直拼贴
    img, meta = OfficialSVGRemixEngine.task_real_flag_splicing("fr", "de", "vertical")
    p = OUTPUT_DIR / "cat8_spliced_01_france_germany_vert.png"
    img.save(p)
    samples.append({"category": "拼接来源", "id": "cat8_01", "img": str(p), "meta": meta})

    # 8.2: 上半部分波兰(白红) + 下半部分乌克兰(蓝黄) 上下水平拼贴
    img, meta = OfficialSVGRemixEngine.task_real_flag_splicing("pl", "ua", "horizontal")
    p = OUTPUT_DIR / "cat8_spliced_02_poland_ukraine_horiz.png"
    img.save(p)
    samples.append({"category": "拼接来源", "id": "cat8_02", "img": str(p), "meta": meta})

    # 8.3: 左半边瑞典(黄色北欧十字) + 右半边瑞士(白色希腊正十字) 左右垂直拼贴
    img, meta = OfficialSVGRemixEngine.task_real_flag_splicing("se", "ch", "vertical")
    p = OUTPUT_DIR / "cat8_spliced_03_sweden_switzerland_hybrid.png"
    img.save(p)
    samples.append({"category": "拼接来源", "id": "cat8_03", "img": str(p), "meta": meta})

    # 同步生成并导出全量真实 JSONL 数据集
    dataset_records = []
    for s in samples:
        gt = GroundTruthSynthesizer.build_structured_gt(s["meta"])
        cap = GroundTruthSynthesizer.build_dense_caption(s["meta"], gt)
        qa = GroundTruthSynthesizer.build_qa_dataset(s["meta"], gt)
        dataset_records.append({
            "id": s["id"],
            "category": s["category"],
            "image": s["img"],
            "gt": gt,
            "caption": cap,
            "qa": qa
        })

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in dataset_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[official_gallery] 官方新画廊 roll 出成功！共 {len(dataset_records)} 条数据，已保存至 {DATASET_PATH}")
    return dataset_records


if __name__ == "__main__":
    generate_official_gallery()
