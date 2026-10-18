# legacy_v1（历史产物，不参与当前训练）

- `gallery/`、`contrastive/`、`density/`、`density_experiments/`、`demo/`、`flywheel_v2/`：v1 演示流水线产物（`generate.py` 的历史四维度）
- `images/`、`images_10k/`：v1 预览图
- `datawheel_10k_captions.parquet`：v1 10,000 条 caption 数据集（仅 250 张预览图，已被 v2 取代）
- `datawheel_dataset.jsonl`、`datawheel_master_dataset.jsonl`、`gallery_dataset.jsonl`：v1 汇总数据
- `DATASET_10K_REPORT.md`：v1 数据报告
- `sft_v1_preview/`：v1 schema 的训练导出回归产物

重建脚本：`tools/generate_10k_parquet.py`（v1 历史）、`generate.py gallery|contrastive|density|demo`。
新数据请使用 `output/datawheel_v2/` 与 `output/datawheel_v6_*/`。
