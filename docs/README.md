# Datawheel 文档索引

| 文档 | 内容 |
|---|---|
| [MID_TRAIN_DATAWHEEL_PLAN.md](MID_TRAIN_DATAWHEEL_PLAN.md) | Qwen3.5-27B 中训练 V6 数据方案：数据种类/任务类型/token 配比、S1/S2/S3 训练影响、远端接入与验收清单 |
| [../README.md](../README.md) | 项目总览：v2 数据集、工具链、快速开始 |
| [../assets/ASSET_CATALOG.md](../assets/ASSET_CATALOG.md) | 资产全景台账 |

## 可视化核验

整理/生成产物后，在项目根目录执行：

```bash
open output/report/index.html
```

HTML 报告包含：数据集总览、样例画廊（带 BBox 标注）、V6 配比与训练影响、质检/校验结果、产物索引与复现命令。

- `output/report/index.html` 为自包含版本（图片以 base64 内嵌，~6MB），可拷到任意目录离线打开；
- `output/report/index_light.html` 为轻量版（~0.4MB，仅适合在项目目录内用 file:// 打开）。

> `output/` 默认被 `.gitignore` 忽略；`output/report/index.html` 由 `tools/build_visual_report.py` 重新生成。
