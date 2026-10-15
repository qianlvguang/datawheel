"""飞轮二期全要素生成与质检验证脚本 (Generate Flywheel v2 Batch)。
生成具备“可控编辑 + 分层标注 + 参考匹配 + 困难对照”的 7 态全矩阵多模态中训练数据集。
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from engine.reference_bank import ReferenceBank
from engine.contrastive_generator import ContrastiveGenerator
from synthesizer.reference_qa import ReferenceQASynthesizer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "flywheel_v2"
IMAGES_DIR = OUTPUT_DIR / "images"
REF_DIR = OUTPUT_DIR / "references"
DEBUG_DIR = OUTPUT_DIR / "visual_debug"
JSONL_PATH = OUTPUT_DIR / "dataset_7_state_sft.jsonl"


def draw_debug_bbox_1000(image: Image.Image, bbox_1000: list, label: str = "") -> Image.Image:
    """在图像副本上绘制相对坐标 0~1000 [ymin, xmin, ymax, xmax] 的标注框与文本。"""
    img_copy = image.copy()
    if bbox_1000 is None:
        return img_copy

    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    ymin, xmin, ymax, xmax = bbox_1000

    px_xmin = int((xmin / 1000.0) * w)
    px_ymin = int((ymin / 1000.0) * h)
    px_xmax = int((xmax / 1000.0) * w)
    px_ymax = int((ymax / 1000.0) * h)

    # 绘制高亮红色/绿色定位矩形
    color = (255, 60, 60) if "mismatch" not in label else (255, 160, 0)
    for offset in range(3):
        draw.rectangle([px_xmin - offset, px_ymin - offset, px_xmax + offset, px_ymax + offset],
                       outline=color)

    # 绘制坐标与标签文本
    tag = f"{label} [{ymin},{xmin},{ymax},{xmax}]"
    draw.rectangle([px_xmin, max(0, px_ymin - 20), px_xmin + len(tag) * 8, px_ymin], fill=(0, 0, 0, 180))
    draw.text((px_xmin + 4, max(0, px_ymin - 18)), tag, fill=(255, 255, 255))

    return img_copy


def run_v2_flywheel_pipeline():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    generator = ContrastiveGenerator(width=640, height=480)
    records = []

    # 选取三大典型实验组：
    # 1. 中国 (五星红旗) vs 越南 (金星红旗) —— 棉质 T 恤褶皱背景
    # 2. 法国 (蓝白红) vs 意大利 (绿白红) —— 陶瓷马克杯曲面背景
    # 3. 欧盟 (十二星环) vs 索马里 (单星) —— 牛皮纸环保包装盒背景
    test_cases = [
        {"ref_id": "cn", "substrate": "tshirt_fabric", "seed": 101},
        {"ref_id": "fr", "substrate": "ceramic_mug", "seed": 202},
        {"ref_id": "eu", "substrate": "kraft_paper", "seed": 303}
    ]

    print("================================================================================")
    print(" 🚀 启动飞轮二期：可控编辑 + 物理融合 + 七态对比矩阵 + 跨图参考匹配数据集生产")
    print("================================================================================")

    for case in test_cases:
        ref_id = case["ref_id"]
        substrate = case["substrate"]
        seed = case["seed"]

        ref_meta = ReferenceBank.get_meta(ref_id)
        print(f"\n[生产目标] 参考对象: {ref_meta['name_zh']} ({ref_id.upper()}) | 底板材质: {substrate}")

        # 1. 渲染并保存标准矢量参考卡片
        ref_card = ReferenceBank.render_reference_card(ref_id, card_size=(400, 300))
        ref_card_path = REF_DIR / f"ref_{ref_id}.png"
        ref_card.save(ref_card_path)

        # 2. 生成 7 态对比样本矩阵
        sample_matrix = generator.generate_7_state_matrix(ref_id, substrate_type=substrate, base_seed=seed)

        for sample in sample_matrix:
            state_key = sample["state_key"]
            img_filename = f"{ref_id}_{state_key.lower()}.jpg"
            img_path = IMAGES_DIR / img_filename
            sample["image"].save(img_path, format="JPEG", quality=90)

            # 生成带 0~1000 BBox 的可视化质检图像
            debug_img = draw_debug_bbox_1000(sample["image"], sample["bbox_1000"], label=sample["match_status"])
            debug_path = DEBUG_DIR / f"debug_{img_filename}"
            debug_img.save(debug_path, format="JPEG", quality=85)

            # 构建符合多模态大模型中训练的跨图问答与接地对话
            qa_record = ReferenceQASynthesizer.build_in_context_qa(
                sample=sample,
                ref_card_path=str(ref_card_path.relative_to(BASE_DIR)),
                query_img_path=str(img_path.relative_to(BASE_DIR))
            )
            records.append(qa_record)

            status_symbol = "✅" if sample["is_positive"] else ("⚠️" if sample["match_status"] == "insufficient_evidence" else "❌")
            print(f"  {status_symbol} 状态 [{state_key:25s}] 判定: {sample['match_status']:20s} 坐标: {str(sample['bbox_1000']):20s}")

    # 保存统一 JSONL 数据集
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n================================================================================")
    print(f"✨ 飞轮二期生产完成！共生成 {len(records)} 条高质量七态多模态训练样本。")
    print(f"📁 训练数据集: {JSONL_PATH}")
    print(f"🖼️ 实物商品图: {IMAGES_DIR}")
    print(f"🎴 标准参考卡: {REF_DIR}")
    print(f"🔍 坐标质检图: {DEBUG_DIR}")
    print("================================================================================")


if __name__ == "__main__":
    run_v2_flywheel_pipeline()
