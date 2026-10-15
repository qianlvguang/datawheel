"""内容与浓度可控阶梯实验生成脚本 (Generate Density Experiments Batch)。
在不引入外部生成模型的前提下，纯靠程序化几何、矢量渲染与物理仿真，合成：
- 3 种不同复杂度的内容实体 (原创积木小怪兽、官方矢量标志、混淆干扰项)
- 5 种不同空间分布与物理印染浓度的阶梯样本 (低空间胸标、中空间复合、满版老花、水洗龟裂做旧、多干扰项密集混淆)
同步产出 0~1000 相对坐标 BBox 质检图与多模态中训练 JSONL 数据集。
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw

from engine.modular_entity import ModularEntityBuilder
from engine.density_engine import ContentAndDensityEngine
from engine.reference_bank import ReferenceBank

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "density_experiments"
IMAGES_DIR = OUTPUT_DIR / "images"
DEBUG_DIR = OUTPUT_DIR / "visual_debug"
ENTITIES_DIR = OUTPUT_DIR / "entities"
JSONL_PATH = OUTPUT_DIR / "dataset_density_sft.jsonl"


def draw_debug_bboxes(image: Image.Image, annotations: List[Dict[str, Any]]) -> Image.Image:
    """在图像副本上绘制所有标注边界框 [ymin, xmin, ymax, xmax] 及详细元数据。"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    for ann in annotations:
        bbox = ann.get("bbox_1000")
        if not bbox:
            continue
        ymin, xmin, ymax, xmax = bbox
        is_target = ann.get("is_target", True)
        label = ann.get("label", "entity")

        px_xmin = int((xmin / 1000.0) * w)
        px_ymin = int((ymin / 1000.0) * h)
        px_xmax = int((xmax / 1000.0) * w)
        px_ymax = int((ymax / 1000.0) * h)

        color = (0, 230, 118) if is_target else (255, 82, 82)
        for offset in range(3):
            draw.rectangle([px_xmin - offset, px_ymin - offset, px_xmax + offset, px_ymax + offset],
                           outline=color)

        tag = f"[{'TARGET' if is_target else 'DISTRACTOR'}] {label} [{ymin},{xmin},{ymax},{xmax}]"
        box_w = len(tag) * 7 + 8
        draw.rectangle([px_xmin, max(0, px_ymin - 20), px_xmin + box_w, px_ymin], fill=(0, 0, 0, 200))
        draw.text((px_xmin + 4, max(0, px_ymin - 17)), tag, fill=(255, 255, 255))

    return img_copy


def run_density_experiment_suite():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    engine = ContentAndDensityEngine(width=640, height=480)
    dataset_records = []

    print("================================================================================")
    print(" 🧪 启动纯程序化内容与浓度阶梯实验 (Zero Generative AI, 100% Code Synthesized)")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # 1. 准备内容库 (Content Preparation)
    # -------------------------------------------------------------------------
    print("\n[内容阶段] 正在构建模块化原创积木实体与高相似度干扰对...")
    # 目标 A: 三眼、双弯角、青色、五星胸徽小怪兽
    target_entity, target_meta = ModularEntityBuilder.build_entity(
        body_type="blob", eye_count=3, accessory="horns", chest_symbol="star", theme_name="cyber_cyan"
    )
    target_path = ENTITIES_DIR / "target_cyan_3eyes_horns.png"
    target_entity.save(target_path)

    # 干扰项 B1: 同青色、单独眼、无角、五星胸徽 (考验眼数与角的细粒度差异)
    distractor_1, d1_meta = ModularEntityBuilder.build_entity(
        body_type="blob", eye_count=1, accessory="none", chest_symbol="star", theme_name="cyber_cyan"
    )
    d1_path = ENTITIES_DIR / "distractor_cyan_1eye_nohorns.png"
    distractor_1.save(d1_path)

    # 干扰项 B2: 同青色、双眼、王冠、五星胸徽 (考验头顶饰物辨析)
    distractor_2, d2_meta = ModularEntityBuilder.build_entity(
        body_type="blob", eye_count=2, accessory="crown", chest_symbol="star", theme_name="cyber_cyan"
    )
    d2_path = ENTITIES_DIR / "distractor_cyan_2eyes_crown.png"
    distractor_2.save(d2_path)

    # 干扰项 B3: 珊瑚粉色、三眼、天线 (考验颜色与配饰多重条件排他)
    distractor_3, d3_meta = ModularEntityBuilder.build_entity(
        body_type="blob", eye_count=3, accessory="antenna", chest_symbol="star", theme_name="coral_pink"
    )
    d3_path = ENTITIES_DIR / "distractor_pink_3eyes_antenna.png"
    distractor_3.save(d3_path)

    print(f"  ✅ 目标实体：{target_meta['discriminative_summary']} (ID: {target_meta['entity_id']})")
    print(f"  ✅ 干扰群：已就绪 3 款细粒度混淆干扰体 (1眼无角、2眼王冠、粉色3眼触角)")

    # -------------------------------------------------------------------------
    # 2. 依次生成 5 种浓度阶梯
    # -------------------------------------------------------------------------
    experiments = [
        {
            "density_id": "exp1_sparse_chest",
            "name": "极简低空间浓度 (胸标 8%~12% 占比)",
            "fn": lambda: engine.render_sparse_chest_motif(target_entity, substrate_type="tshirt_fabric", seed=201),
            "substrate": "纯棉卫衣浅燕麦底",
            "prompt_task": "在商品图上定位左胸口的原创微型印花，并确认其眼数与饰物特征。"
        },
        {
            "density_id": "exp2_medium_compound",
            "name": "标准中空间浓度 (中心大幅印花 35% 占比)",
            "fn": lambda: engine.render_medium_compound(target_entity, substrate_type="tshirt_fabric", seed=202),
            "substrate": "纯棉 T 恤大幅印花",
            "prompt_task": "分析中心复合印花的主体，判断是否与目标三眼怪兽一致。"
        },
        {
            "density_id": "exp3_dense_monogram",
            "name": "满版极高空间浓度 (老花重复阵列 >70% 占比)",
            "fn": lambda: engine.render_dense_monogram_allover(target_entity, substrate_type="kraft_paper", tile_size=75, seed=203),
            "substrate": "牛皮纸包装箱满印",
            "prompt_task": "分析整幅满版重复平铺的老花图案，识别构成该图案的核心基元实体。"
        },
        {
            "density_id": "exp4_washed_vintage_fade",
            "name": "稀释做旧物理浓度 (50% 透底水洗 + 龟裂磨损)",
            "fn": lambda: engine.render_washed_vintage_fade(target_entity, substrate_type="tshirt_fabric", opacity=0.50, seed=204),
            "substrate": "复古水洗做旧棉织布",
            "prompt_task": "印花出现严重掉色透底与表面龟裂，请判定残存关键轮廓是否仍符合目标实体。"
        },
        {
            "density_id": "exp5_cluttered_multi_distractors",
            "name": "多干扰项密集混淆浓度 (4 槽位多目标排他定位)",
            "fn": lambda: engine.render_cluttered_multi_distractors(target_entity, [distractor_1, distractor_2, distractor_3], substrate_type="ceramic_mug", seed=205),
            "substrate": "陶瓷杯柱面曲面",
            "prompt_task": "商品表面印有多只相似怪兽，请排除单眼、双眼及粉色怪兽，唯一定位【三眼、带双弯角】的青色怪兽。"
        }
    ]

    print("\n[浓度阶段] 开始执行 5 梯度渲染与密集标注追踪...")

    for exp in experiments:
        did = exp["density_id"]
        dname = exp["name"]
        print(f"\n▶ 正在生成: [{did}] {dname} | 底材: {exp['substrate']}")

        img, annotations = exp["fn"]()

        img_filename = f"{did}.jpg"
        img_path = IMAGES_DIR / img_filename
        img.save(img_path, format="JPEG", quality=90)

        # 绘制标注质检图
        debug_img = draw_debug_bboxes(img, annotations)
        debug_path = DEBUG_DIR / f"debug_{img_filename}"
        debug_img.save(debug_path, format="JPEG", quality=85)

        # 构造多模态对话数据
        primary_ann = [a for a in annotations if a.get("is_target", True)][0]
        target_box = primary_ann.get("bbox_1000")

        assistant_answer = {
            "experiment": did,
            "density_level": dname,
            "target_detected": True,
            "primary_target_bbox_1000": target_box,
            "discriminative_verification": {
                "eye_count": target_meta["eye_count"],
                "accessory": target_meta["accessory_desc"],
                "body_style": target_meta["body_desc"],
                "chest_symbol": target_meta["chest_desc"]
            },
            "total_objects_in_scene": len(annotations),
            "scene_description": f"在{exp['substrate']}上成功完成{dname}维度的可控定位。"
        }

        record = {
            "id": did,
            "density_type": did,
            "image": str(img_path.relative_to(BASE_DIR)),
            "conversations": [
                {
                    "role": "user",
                    "content": f"<|image|>\n{exp['prompt_task']}\n请给出目标实体的 0~1000 相对定位坐标 [ymin, xmin, ymax, xmax] 并列举判定依据。"
                },
                {
                    "role": "assistant",
                    "content": f"```json\n{json.dumps(assistant_answer, ensure_ascii=False, indent=2)}\n```\n\n**定位结论**：已在目标区域 `{target_box}` 检测到符合【{target_meta['discriminative_summary']}】的实体。"
                }
            ],
            "annotations": annotations
        }
        dataset_records.append(record)
        print(f"  🎯 目标坐标: {target_box} | 场景目标数: {len(annotations)}")

    # 保存 JSONL 数据集
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in dataset_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n================================================================================")
    print(f"✨ 阶梯实验全部完成！已产出 {len(dataset_records)} 种不同内容与浓度阶梯的高质量多模态样本。")
    print(f"📁 训练数据 JSONL: {JSONL_PATH}")
    print(f"🖼️ 实物商品图目录: {IMAGES_DIR}")
    print(f"🔍 坐标质检图目录: {DEBUG_DIR}")
    print(f"👾 原创积木实体库: {ENTITIES_DIR}")
    print("================================================================================")


if __name__ == "__main__":
    run_density_experiment_suite()
