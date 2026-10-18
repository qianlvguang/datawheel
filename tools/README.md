# tools/ 工具索引

> 所有脚本都在项目根目录执行：`.venv/bin/python tools/<script>.py`。
> 脚本通过 `Path(__file__).parent.parent` 定位项目根，**新增工具请保持放在 tools/ 平级目录**（移动会改变相对路径）。

## 资产构建 / 台账

| 工具 | 作用 |
|---|---|
| `build_asset_index.py` | 982 个前景图元视觉索引（主导色/几何/dHash 近邻） |
| `build_substrate_quality.py` | 1,500 张真实底板可印性/难度/区位索引（447 张合格） |
| `build_entity_name_table.py` | 582 条官方实体中英名表（271 国旗 / 121 国徽 / 190 地方旗） |
| `verify_entity_names.py` | 官方实体中文命名覆盖率校验（报告写到 `output/_debug/`） |
| `asset_catalog_manager.py` | 资产台账巡检总控（含 v1 legacy 10k caption 数据集统计） |
| `asset_fetcher.py` / `batch_asset_downloader.py` / `download_*.py` / `assemble_multi_anime_assets.py` / `extract_1500_substrates.py` | 历史一次性素材下载与整理（联网脚本；结果已在 assets/，一般不需要重跑） |

## v2 主数据集（10,000 条）

| 工具 | 作用 |
|---|---|
| `generate_datawheel_v2.py` | v2 五阶段 10k 生成器（分片/渲染/合并/报告/contact sheet） |
| `flywheel_audit.py` | 数据集质检审计（critical=0 门禁） |
| `export_training_data.py` | ShareGPT / ms-swift 分 split 导出 |
| `generate_10k_parquet.py` | **v1 历史** 10k caption parquet 生成（产物在 `output/legacy_v1/`） |

## V6 中训练数据（Qwen3.5-27B）

| 工具 | 作用 |
|---|---|
| `build_v6_single_image.py` | 试点：单图多轮 QA / 结构化 JSON / 文本一致性（1,850 行） |
| `build_v6_reference_rows.py` | 试点：多图参考匹配（650 行） |
| `pack_v6_pilot.py` | 合并试点 parts、统计 token、生成远端上传包 |
| `validate_swift_v6.py` | V6 ms-swift 契约校验（schema / 图片 / 长度 / cate 分布） |
| `build_v6_stage_mixtures.py` | light 档 S1/S2/S3 stage 文件（重复 r=3/2/4） |
| `build_v6_balanced_single.py` | balanced：单图多任务 37,186 行 |
| `build_v6_balanced_reference.py` | balanced：多图参考匹配 6,707 行 |
| `assemble_v6_balanced.py` | union 43,893 → S1/S2/S3 stage 文件 + validator + 上传包 |

## 数据语言清理

| 工具 | 作用 |
|---|---|
| `normalize_dataset_language.py` | 清理程序化/矢量/模式名等实现术语；重写 Stage A 的"区域名带颜色 + 又描述颜色"冗余；去掉 caption 里的场景括注、知识扩写句和 bbox；压缩拼接式品类名；V6 结构化 caption JSON 同步清理 |

> v2 全量生成后由 `generate.py v2` 自动调用；V6 parts 构建后执行
> `.venv/bin/python tools/normalize_dataset_language.py --skip-v2`，再 pack/assemble。

## 可视化

| 工具 | 作用 |
|---|---|
| `build_visual_report.py` | 生成 `output/report/index.html`（自包含，图片 base64 内嵌）/ `index_light.html`：数据集总览 / 样例画廊（BBox 标注）/ 配比与质检 / 产物索引 |

## 约定

- 产物统一写 `output/`（已被 `.gitignore` 忽略）；历史 v1 产物在 `output/legacy_v1/`。
- 代码路径优先 `PROJECT_ROOT / ...`，禁止写死本机绝对路径。
- 修改数据生成逻辑后，请同步重跑 `validate_swift_v6.py` / `flywheel_audit.py` 并更新 docs。
