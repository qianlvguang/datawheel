# 自动图案摆放方案（Placement Optimizer / Visual Loop）

> 分支：`assist`（不合并/不推送 `main`）
> 背景：生成式编辑模型倾向于把图案放在画面中心，不理解商品结构（衣身、包盖、帽冠、鞋侧、可印区、褶皱/遮挡）。
> 目标：用**结构化定位 + 确定性渲染 + 视觉评分闭环**替代"让模型自己猜位置"，同时给 v4.1 flash 一类 agent 一套强制看图的工具契约。
>
> **实现状态（assist 分支，2026-10-18）**：M1 审计 + M2 确定性优化器已实现并跑通；
> M3（新商品图分割）/ M4（Qwen 遮罩回退）/ M5（agent 工具队列）待实现。

---

## 1. 问题定义

当前失败模式：

1. 输入商品图 + 图案，要求模型"把图案放到商品上"；
2. 模型没有 3D/服装结构先验，默认选择**图像几何中心**；
3. 生成的图案可能压到接缝、口袋、拉链、印花禁区、背景或商品外；
4. 后续 bbox/QA 标签与真实位置不一致，数据被污染。

核心判断：**"放在哪里"不应由语言/图像生成模型决定，应由商品结构解析 + 可量化评分决定。**
模型只负责两件事：
- 在候选布局中做高层选择/解释（可选）；
- 在 mask 约束下做最终融合（可选，失败可回退）。

---

## 2. 总体架构

```text
                    ┌──────────────────────────┐
商品图 ────────────▶│ 1. 结构解析 / 可印区推断  │
                    └────────────┬─────────────┘
                                 │ zone polygon / mask / category / view
                                 ▼
                    ┌──────────────────────────┐
                    │ 2. 候选布局生成          │
                    │ 位置/尺度/旋转/透视/褶皱 │
                    └────────────┬─────────────┘
                                 │ N candidates
                                 ▼
                    ┌──────────────────────────┐
                    │ 3. 确定性渲染            │
                    │ engine/compositor.py     │
                    └────────────┬─────────────┘
                                 │ render + design mask
                                 ▼
                    ┌──────────────────────────┐
                    │ 4. 视觉评分器            │
                    │ containment/SSIM/面积/溢出│
                    └────────────┬─────────────┘
                        ┌────────┴────────┐
                        ▼                 ▼
                   通过：输出最佳      不通过：局部细化/回退
                        │                 │
                        │                 ├─▶ 5a 调整候选参数重试 2~3 轮
                        │                 └─▶ 5b Qwen mask/标注编辑（可选）
                        ▼
               placement_manifest.jsonl + 渲染图 + 标签
```

**优先级：** 确定性合成 > Qwen mask 编辑 > 自由生成。
自由生成只用于生成空白商品底图/背景，不用于决定图案位置。

---

## 3. 结构解析与可印区推断

### 3.1 已有元数据优先（P0）

- 数据源：`assets/manifests/substrate_quality.json` / `output/datawheel_v2/dataset.parquet` 中的 `substrate_id`、`placement_zone`、`zone_canvas_norm`；
- 输入：商品图 + 底板 ID；
- 输出：`zone_polygon`（0~1000 坐标）、`zone_mask`、类别/视角；
- 适用：现有 v2 真实底板（447 张合格底板），可直接复用。

### 3.2 新商品图：品类模板 + 分割（P1）

| 品类 | 默认可印区（归一化） | 备注 |
|---|---|---|
| T恤/卫衣（正面） | 前胸中央偏上：x 30%~70%，y 18%~48% | 避开领口/下摆 |
| 箱包（正面） | 包身前面板：x 25%~75%，y 25%~75% | 避开拉链/口袋 |
| 棒球帽（正面） | 帽冠前面：x 30%~70%，y 25%~65% | 避开帽檐 |
| 运动鞋（侧面） | 鞋帮外侧：x 35%~75%，y 35%~70% | 避开鞋带/鞋底 |
| 连衣裙/上衣 | 前胸或下摆，按品类选择 | 依赖款式 |

实现顺序：

1. 分割商品主体：优先 `rembg`（轻量、CPU 可跑）；有 GPU 时用 SAM2 / GroundingDINO + SAM；
2. 用商品 mask 的 bbox + 品类模板生成候选 zone；
3. 对 zone 做合理性检查：面积占比、是否贴边、是否在 mask 内、是否与接缝/拉链冲突（如有 edge/line 检测）；
4. 无 mask/无品类时：回退到该底板的已有 zone；仍无 → 标记 `needs_manual_zone`，人工只标一次即可复用。

### 3.3 人工 zone 标注（只做一次）

- 每类商品/视角标 5~20 张，生成 `zone_templates.json`；
- 后续商品图按类别 + 相似度匹配模板；
- 人工成本低，且 zone 可复用、可审计。

---

## 4. 候选布局生成

### 4.1 参数化

每个候选 = 设计图矩形 → 目标四边形（homography）：

```python
Candidate = {
    "candidate_id": "c0001",
    "center_xy": (0.5, 0.35),      # 相对 zone bbox
    "scale": 0.8,                  # 相对 zone 宽/高
    "rotation_deg": -6.0,
    "perspective": {"tl": (-4, 2), "tr": (3, -2), "bl": (2, 4), "br": (-3, -3)},
    "fold_strength": 0.15,         # 面料褶皱/弯曲
}
```

### 4.2 搜索策略

1. **粗采样**：96~128 个候选（位置/缩放/旋转/四角透视扰动）；
2. **过滤**：设计 mask 必须完全落在 zone/product mask 内，面积比在品类范围；
3. **精修**：对 top 5~10 做局部坐标下降/CMA-ES 风格扰动（每轮 16~32 个）；
4. **停止**：达到阈值（如 score ≥ 0.82）或预算耗尽；
5. 输出 top-3，默认取最高分。

全部使用 `engine/compositor.py` 渲染，**不调用扩散模型**，单图毫秒级。

---

## 5. 视觉评分器

### 5.1 指标

| 指标 | 定义 | 建议阈值 |
|---|---|---|
| `inside_zone` | 设计像素 ∩ zone mask / 设计像素 | ≥ 0.98 |
| `inside_product` | 设计像素 ∩ product mask / 设计像素 | ≥ 0.99 |
| `area_ratio` | 设计面积 / zone 面积 | 品类区间，如 0.05~0.6 |
| `design_ssim` | 渲染后设计 vs 源设计（透视归一化） | ≥ 0.85 |
| `spill_ratio` | 设计落在 zone 外的比例 | ≤ 0.02 |
| `center_offset` | 设计中心距 zone 中心距离 | 小于 zone 短边的 0.15 |
| `edge_clearance` | 设计 bbox 到 zone 边界最小距离 | ≥ 2% zone 短边 |
| `visibility` | 设计区域对比度/清晰度 | 按底板质量，≥ 阈值 |

加权总分：

```text
score = 0.30*inside_zone + 0.15*inside_product + 0.15*area_score
      + 0.15*design_ssim + 0.10*spill_score + 0.10*center_score
      + 0.05*visibility_score
```

### 5.2 可选 VLM 复核

- 只对 `score` 处于边界（0.7~0.82）或人工队列样本调用 VLM；
- 让 VLM 判断"图案是否贴合商品结构、是否在合理穿着位置"，输出 pass/fail + 理由；
- VLM 的结论不能覆盖硬指标；只用于排序/升级人工。

---

## 6. Qwen-Image-Edit 的可控回退

只有在确定性合成无法达到效果（例如需要真实褶皱/遮挡/材质融合）时才使用：

1. 用优化器选出最佳 zone/候选，在商品图上画**目标框/遮罩**；
2. 调 `tools/qwen_image_ops.py edit`：
   - `image` = 商品图；
   - `mask_image` = 只允许可印区/目标框重绘；
   - prompt 明确 "place design exactly inside the highlighted region; keep product geometry unchanged"；
3. 生成后用第 5 节评分器重算；同时对比**保护区域**（商品/非目标区域）像素差；
4. 不通过 → 回退确定性合成或进入人工；通过 → 记录编辑原因与参数；
5. 若编辑改变了商品几何，不用该图做正样本（或必须重算 homography 与 bbox）。

---

## 7. 给 v4.1 flash / agent 的工具契约

目标：让模型**不能跳过看图**，也不能自由给坐标。

### 7.1 工具

```text
inspect_product(image_path)
  -> {product_type, view, confidence, product_mask_path,
      printable_zones: [{zone_id, polygon, score}], warnings}

list_candidates(image_path, design_path, zone_id)
  -> [{candidate_id, transform, predicted_score, render_preview_path}]

score_candidate(candidate_id, design_path, zone_id)
  -> {score, metrics: {inside_zone, inside_product, area_ratio, design_ssim, spill_ratio}}

commit_placement(sample_id, candidate_id)
  -> {status, output_image, output_bbox, manifest_path}
```

### 7.2 强制流程

- 每个样本必须先调用 `inspect_product`，否则 `commit_placement` 拒绝；
- 只能从 `list_candidates` 返回的 ID 中选择，禁止模型直接输出坐标；
- 最终必须返回 JSON：

```json
{
  "sample_id": "DW_V2_001002",
  "candidate_id": "c0007",
  "verdict": "pass",
  "evidence": {
    "inside_zone": 0.995,
    "design_ssim": 0.91,
    "reason": "前胸可印区内，未压到领口/下摆"
  }
}
```

- 程序校验：
  - 必须有 inspect 记录；
  - candidate_id 必须存在；
  - verdict=pass 时 metrics 必须过阈值；
  - 缺字段/缺调用 → 自动重试该样本，2 次失败进人工队列；
- 这样 flash 即使"懒"，也无法跳过工具链。

### 7.3 批量执行

- 外层 Python 脚本逐条发请求（或 10~20 条小批），每条强制带 `sample_id`；
- 返回条数 ≠ 输入数量 → 只重试缺失 ID；
- 记录完成率、pass 率、人工升级率；完成率低的模型不能进生产门禁。

---

## 8. 与 Datawheel 的接入

### 8.1 Phase 1：审计现有数据（不需要新图）

- 新增 `tools/placement_audit.py`：
  - 读取 v2 dataset；
  - 对有 bbox 的样本计算"图案中心 vs 图像中心"距离、zone 包含率、面积比；
  - 输出 `center_collapse_rate`（中心塌缩率）和问题清单；
  - 对 Stage A 合成色块不适用（其设计就是几何中心），按 stage 排除。
- 产出：`output/audit/placement_audit.json` + 可视化叠加图。

### 8.2 Phase 2：优化器原型

- 新增 `tools/placement_optimizer.py`：
  - 输入：商品图、设计图、product_type/zone（可选）；
  - 输出：最佳渲染图、候选 manifest、叠加 bbox 的 QA 图；
  - 依赖：`engine/compositor.py`、`engine/coords.py`，不依赖新模型。
- 测试数据：现有 447 张合格底板的可印区 + 随机图案；无需真实用户图。

### 8.3 Phase 3：接入 Qwen 空白商品

- 流程：Qwen 生成空白商品 → 分割/zone 推断 → placement_optimizer → 合成 → 现有 QA/caption 生成；
- 生成数据只进 train split，比例 ≤20~30%；
- 每个样本记录 `source_type=qwen_gen`、`zone_source`、`placement_score`。

### 8.4 Phase 4：agent 接入

- 将 7.1 工具包装成 HTTP/JSON 或 Python callable；
- v4.1 flash 只做候选排序/解释；程序做最终判定；
- 人工队列只处理分类为 `uncertain`/`reject` 的样本。

---

## 9. 质量门禁与验收

### 9.1 必须指标

| 指标 | 目标 |
|---|---|
| 中心塌缩率（非中心品类） | < 2% |
| zone 包含率中位数 | ≥ 0.98 |
| 设计 SSIM 中位数 | ≥ 0.88 |
| 溢出率（spill > 0.02） | < 3% |
| 人工抽检通过率 | ≥ 95% |
| agent 工具链完成率 | ≥ 99% |
| 编辑后保护区域异常率 | < 2% |

### 9.2 缺陷注入测试

准备 100 条 golden set：
- 图案故意放中心/贴边/压拉链/超出商品；
- 设计被拉伸/旋转错；
- Qwen 编辑后商品被改坏；
- 无 zone/低质量底板。

要求 placement 评分器对已知缺陷召回率 ≥ 95%，否则调阈值/加特征，不进入生产。

---

## 10. 实施里程碑

| 里程碑 | 内容 | 产出 | 依赖 |
|---|---|---|---|
| M1 | 现有数据摆放审计 | `placement_audit.py` + 报告 | 现有 dataset |
| M2 | 确定性优化器 | `placement_optimizer.py` + 50 张底板测试 | compositor、zone 元数据 |
| M3 | 新商品图 zone 推断 | 品类模板 + rembg/SAM 适配 | 可选 GPU |
| M4 | Qwen 回退链路 | mask/标注编辑 + 保护校验 | Qwen endpoint |
| M5 | agent 工具契约 | inspect/list/score/commit + 队列 | v4.1 flash API |
| M6 | 数据接入与验证 | train-only 新数据 + V6 重建 + 审计 | M1-M5 |

**M1、M2 不依赖你提供图片**：用现有底板和 zone 元数据即可先跑通；M3 再补真实商品图。

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| 商品 mask/zone 不准 | 置信度阈值 + 人工只标模板；低置信度进队列 |
| 多视角/平铺 vs 模特图 | 按 view 分类模板；先只做正面/平铺 |
| 透视/褶皱不自然 | compositor homography + 局部变形；必要时 Qwen 融合 |
| Qwen 改坏商品 | protect bbox + 像素差 + 模板 SSIM；失败回退 |
| agent 偷懒/跳过检查 | 工具强制调用 + 返回条数校验 + 缺项重试 |
| 生成数据分布偏移 | ≤20~30% 比例；按品类/来源分别评测 |
| 许可 | Qwen-Image-2.1 研究许可；商用需确认 |

---

## 12. 建议新增文件

```text
tools/placement_audit.py          # M1：现有数据摆放审计
tools/placement_optimizer.py      # M2：候选生成 + 评分 + 渲染
tools/placement_tools.py          # M5：给 agent 的 inspect/list/score/commit
tools/zone_templates.py           # 品类 zone 模板 + 相似度匹配
assets/manifests/zone_templates.json
docs/PLACEMENT_OPTIMIZER_PLAN.md  # 本文件
```

---

## 13. 已实现脚本（M1 / M2）

| 文件 | 状态 | 作用 |
|---|---|---|
| `tools/placement_common.py` | ✅ | 品类 zone 模板、substrate 加载、letterbox/可印区坐标换算 |
| `tools/placement_audit.py` | ✅ | 现有 v2 数据摆放审计（bbox vs 可印区、中心塌缩率、issues + 叠加图） |
| `tools/placement_optimizer.py` | ✅ | 候选布局采样 + compositor 渲染 + 评分 + 局部精修 |

### 13.1 M1 实测

```bash
.venv/bin/python tools/placement_audit.py \
  --dataset output/datawheel_v2/dataset.parquet \
  --out output/audit/placement
```

当前 v2 结果：

| 指标 | 数值 |
|---|---:|
| 审计 bbox | 9,296 |
| center collapse | 82 / 9,296 (0.88%) |
| zone containment 中位数 | 1.0 |
| Stage B / C / E 中心塌缩率 | 0 / 1.64% / 0 |

说明：Stage C 的 `placement_zone` 是组合布局 ID，审计回退到整体可印区中心，C 的 82 个 flag 主要来自布局中心与图像中心接近的合法情况，适合作为人工复核队列而不是硬失败。

### 13.2 M2 实测

```bash
# T 恤前胸
.venv/bin/python tools/placement_optimizer.py \
  --substrate-id comm_studio_topwear_19457 \
  --zone-id center_chest_graphic \
  --design assets/official_flags/fr.svg \
  --out output/placement/demo_topwear \
  --candidates 96 --refine-rounds 2 --refine-samples 24
```

| 品类 | score | inside_zone | center_dist | 说明 |
|---|---:|---:|---:|---|
| T 恤（center_chest_graphic） | 0.995 | 1.0 | 0.003 | 图案落在前胸可印区，不贴边 |
| 包（bag_front_panel） | 0.985 | 1.0 | 0.013 | 包身正面 |
| 鞋（shoe_lateral_quarter） | 0.980 | 1.0 | 0.024 | 鞋帮侧面 |
| 帽（hat_front_panel） | 0.989 | 1.0 | 0.012 | 帽冠正面 |

单次 96+48 候选约 2 秒；不调用扩散模型，确定性可复现。

输出：
- `best_render.jpg`：最佳渲染
- `debug.jpg`：蓝框=可印区、红框=图案 bbox、红线=图像中心
- `placement_metrics.json`：最佳参数/评分/bbox
- `candidates.jsonl`：前若干候选

### 13.3 已知限制

- 目前只对已有 `substrate_quality.json` 可印区的商品图直接可用；
- 任意新商品图仍需要 M3 的分割/品类 zone 推断（或人工标一次 zone）；
- M2 没有调用 Qwen 做材质融合，极端褶皱/深色面料仍可能有贴纸感；M4 再补遮罩回退。

## 14. 验收命令（规划）

```bash
# M1 审计
.venv/bin/python tools/placement_audit.py \
  --dataset output/datawheel_v2/dataset.parquet \
  --out output/audit/placement

# M2 单图优化
.venv/bin/python tools/placement_optimizer.py \
  --image output/datawheel_v2/images/DW_V2_001002.jpg \
  --design assets/brand_logos/netflix.svg \
  --product-type tshirt --zone auto \
  --out output/placement/DW_V2_001002

# M5 agent 批量
.venv/bin/python tools/placement_tools.py \
  --plan output/review/placement_plan.jsonl \
  --api-base http://<flash-endpoint>/v1 \
  --out output/review/placement_results.jsonl
```

---

## 15. 最终判断

**不要让模型"自己看图找位置"**。
正确的自迭代是：

```text
结构解析 → 候选布局 → 确定性渲染 → 可量化评分 → 局部参数搜索
```

模型/agent 只在候选之间做选择、解释、抽样复核。
这样即使 v4.1 flash 很懒，也不会把图案放到画面中心。
