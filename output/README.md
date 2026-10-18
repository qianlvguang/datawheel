# output/ 产物索引

> 本目录中的大数据/图片被 `.gitignore` 忽略，只保留在本地；可用下列脚本重建。
> 打开 `report/index.html`（双击或 `open output/report/index.html`）可本地核验样例与 BBox。

```
output/
├── report/index.html          # 🖥️ 本地核验可视化（自包含 ~6MB，图片内嵌，任意位置可开）
├── report/index_light.html    # 轻量版（~0.4MB，图片用项目相对路径）
├── datawheel_v2/              # 主线一：10,000 条全量渲染多模态数据集
│   ├── dataset.parquet        # 10,000 行 × 35 列（zstd）
│   ├── images/                # 10,000 张 JPEG（640×640 / 640×840）
│   ├── report.md              # 数据统计报告
│   ├── contact_sheet.jpg      # 20 图拼图
│   ├── audit/                 # flywheel_audit 三件套（critical=0）
│   ├── sft/                   # ShareGPT / ms-swift 训练导出（5 split × 2 格式）
│   └── _generation/           # 分片中间件与生成日志（不再参与训练）
├── datawheel_v6_pilot/        # 主线二·试点：2,500 行 V6 中训练数据
│   ├── dataset.parquet, images/, upload/, stage_files/, parts/
│   ├── validation/            # ms-swift 契约校验（critical=0）
│   ├── mix_analysis.md, mix_plan.json, pilot_manifest.json, pilot_stats.md
├── datawheel_v6_balanced/     # 主线二·扩规模：union 43,893 行
│   ├── parts/                 # 单图/多图生成 parts
│   ├── stage_files/           # S1/S2/S3 parquet + balanced_manifest.json + validation/
│   ├── upload/                # 远端上传包（s1/s2/s3 + 图片 + README）
│   └── images/                # query 裁图与参考图渲染
├── legacy_v1/                 # 历史 v1 演示产物（gallery/contrastive/density/demo、10k caption parquet 等）
└── _debug/                    # 调试图与命名覆盖报告
```

## 复现命令

```bash
# v2 10k 数据集
.venv/bin/python generate.py v2

# V6 试点
.venv/bin/python tools/build_v6_single_image.py
.venv/bin/python tools/build_v6_reference_rows.py --limit 650
.venv/bin/python tools/pack_v6_pilot.py --upload
.venv/bin/python tools/validate_swift_v6.py --dataset output/datawheel_v6_pilot/dataset.parquet

# V6 balanced
.venv/bin/python tools/build_v6_balanced_single.py
.venv/bin/python tools/build_v6_balanced_reference.py
.venv/bin/python tools/assemble_v6_balanced.py --upload

# 可视化核验
.venv/bin/python tools/build_visual_report.py
open output/report/index.html
```
