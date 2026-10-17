# Datawheel V6 × Qwen3.5-27B 中训练数据方案（v0.2）

> 目标：把 Datawheel 的程序化合成数据作为 **V6 数据通道**接入 Qwen3.5-27B 的 S1/S2/S3 中训练，
> 在不破坏现有 V1–V5 配比与防遗忘策略的前提下，补齐「细粒度视觉证据」能力：
> 小目标定位、多图参考匹配、相似干扰排除、证据不足拒绝、结构化视觉描述、视觉事实一致性。
>
> 本文件是**方案主文档**；试点数据、校验报告与配比计算在 `output/datawheel_v6_pilot/`（本地，不入 git）。
> 训练侧事实来源：`~/Documents/Coding/Mid_train/recommended_pipeline.md`、`进展日志.md`。
> 更新：2026-10-18，试点集已生成并通过 ms-swift 契约校验（2,500 行 / 3,225 图）。

---

## 1. 训练侧约束（来自 Mid_train 实测方案）

| 项 | 现状 |
|---|---|
| 底座 | Qwen3.5-27B（dense VLM），累计 LoRA |
| 框架 | ms-swift 4.3.2 + Megatron-Core 0.18.0，H200×8，TP4/DP2 |
| 数据格式 | parquet 列 `messages, images, enable_thinking, sample_id, cate_id, variant` |
| 通道 | 每份 parquet 独立传入 `--dataset`；插件把 `variant` 映射为 `channel=V{n}`，loss 记 `loss_V1..V5` |
| 长度 | MAX_LENGTH=8192，packing=true，GBS=32 packed windows，strict 不静默丢行 |
| S1 | V1–V4 mini 800k + V5 60k = 860k rows；2 epochs；84,217 windows；ipe≈2,632 |
| S2 | V1–V4 medium 540k + V5 rough 60k = 600k rows；视觉冻结；2 epochs；58,936 windows；ipe≈1,842 |
| S3 | V1–V4 final 1,097,530 + V1 replay 120,000；1 epoch；≈119,366 windows；ipe≈3,730（**是否含 V5 60k 待确认**） |
| 防遗忘 | S3 有 V1 回放 120k；V5 为 S1-seen replay，禁止切验证集 |
| 行级键 | `(sample_id, cate_id, images)` 唯一；禁止按 (task,sid) 去重 |
| 图片路径 | 相对 `/data/autovision-cfs/autovision-notebook`；`IMAGE_MAX_TOKEN_NUM=16384` |

**结论：V6 无需改训练插件。** `variant=6` 会被现有插件自动映射为 `loss_V6`，
接入方式与 V4/V5 相同：上传 parquet + 图片 → 在对应阶段脚本的 `TRAIN_FILES` 追加一行 → smoke 确认无丢行后正式跑。

---

## 2. V6 数据定位

27B 中训练的目标是**可迁移的视觉证据能力**，不是刷业务答案分布。因此：

- **强化**：定位（bbox）、多图参考身份匹配、相似干扰排除、证据不足/拒绝、条件计数与空间关系、
  结构化视觉描述、视觉事实与文字描述的一致性判断。
- **压低**：纯色块几何（A 阶段）对 27B 价值低，V6 全量中占比 ≤3%，仅作通道校准。
- **不做**：业务规则背诵、V5 治理规则复刻、长 CoT thinking。
- **注意 caption 稀释效应**：Cambrian-1 等工作的经验是"普通 caption 占比过高会稀释精确实体监督"。
  因此 V6 的 structured_caption 只放在 S2（35% 行占比），S1/S3 以 grounding/参考匹配/负样本为主；
  S2 的 JSON 占比与格式锁定风险要在评测中单独观察。

### 2.1 数据种类（6 类）与 `cate_id`

| 种类 | `cate_id` | 内容 | 对应能力/评测簇（待与 eval_v3 task 表对账） |
|---|---|---|---|
| 单图 grounding/属性 | `apparel` / `flag-hard` / `ip-brand` / `anime-ip` / `quantity` | 单图 2~3 轮 QA：识别/bbox/颜色/位置/计数/方位 | apparel、flag-hard、vision-strong、IP |
| 多图参考匹配 | `reference_match` | 1 张商品图 + 2~3 张候选参考图（正例/困难负例混合），判断匹配并给 bbox | misleading、IP、flag-hard |
| 负样本/证据不足 | `misleading` | 不存在、被遮挡、相似干扰；答"否/无法确认" | misleading |
| 结构化描述 | `structured_caption` | 严格 JSON：品类/工艺位置/元素/bbox/颜色/描述 | vision-strong、apparel、text-compliance |
| 文本一致性 | `text-compliance` | 判断描述与图像是否一致（约一半故意篡改区位/颜色/bbox/工艺） | text-compliance |
| 几何基元（少量） | `geometry-calibration` | 极简色块/条纹，用于通道 loss 稳定性对照 | 基础视觉 |

### 2.2 试点集构成与实测结果（`output/datawheel_v6_pilot/`）

| 部分 | 行数 | cate 分布 | 说明 |
|---|---:|---|---|
| `single_image.parquet` | 1,850 | 6 类各 200 + structured_caption 400 + text-compliance 250 | 单图多轮 QA（2~3 轮）/ 结构化 JSON / 一致性判断 |
| `reference_multi.parquet` | 650 | reference_match 650（476 match / 174 no_match） | 1 商品图 + 2~3 候选参考图（3~4 图/行） |
| **dataset.parquet** | **2,500** | 9 个 cate | 3,225 张唯一图片、0 缺失 |

实测指标：

- 每行图片数：1 张 1,850 行 / 3 张 269 行 / 4 张 381 行；两轮对话行占 39.7%；
- 文本 token 估算合计 200,146（mean 80.1/行）；
- token 估算：**512 图/张口径 2,340,818（≈936 tok/行，≈288 windows）**；
  **1024 图/张保守口径 4,481,490（≈1,793 tok/行，≈552 windows）**；
- 单行最长保守估算 4,188 tokens（1024/图），**无超长行**；
- `validate_swift_v6.py`：critical=0、exit 0；唯一 warning 是多图行共享候选参考图（预期）；
- split 策略：**只使用 v2 `train` split**（stage_d 也只取 train），不泄入 v2 的 val / test_unseen_* 泛化评测集。

---

## 3. 数据契约（与 V1–V5 完全一致）

```text
columns:
  messages          list<struct<role: string, content: string>>   # user/assistant 交替
  images            list<string>    # 相对 /data/autovision-cfs/autovision-notebook
  enable_thinking   bool            # V6 全部 False
  sample_id         string          # V6S_* / V6R_*，全局唯一
  cate_id           string          # 见 2.1
  variant           int32           # = 6
行级唯一键: (sample_id, cate_id, images)
```

- `<image>` 占位符数量 = `len(images)`，放在第一条 user 消息里；
- 多图样本 images 顺序与 `<image>` 顺序一致（第 1 张商品图、后续候选参考图）；
- assistant 答案短、可验证；不引入 thinking；
- 多图行控制 ≤4 张图（1 商品 + ≤3 参考），按 1024/图最坏估算仍 <8192；
- **任何超过 8192 的行都不许静默截断**；远端 smoke 必须用 `strict=true` 暴露。

---

## 4. 配比方案（S1 / S2 / S3）

### 4.1 计量口径

- 行数≠算力。以 **token 份额**为准：`V6 token 占比 = V6 tokens / 该阶段每 epoch 总 tokens`。
- 现有阶段 token 预算（Mid_train 实测 windows × 8192）：
  - S1: 84,217 × 8192 ≈ **690M tokens/epoch**（实测 8104 tok/window → 682.5M）
  - S2: 58,936 × 8192 ≈ **483M tokens/epoch**（实测 8114 → 478.2M）
  - S3: ≈119,366 windows × 8192 ≈ **978M tokens/epoch**（估算，待实测）
- V6 行 token 估算：文本 = 中文汉字/1.6 + 英文数字/4；图片按 512 tokens/张（另给 256/1024 敏感性）；
  packing 效率按 8104~8114 tok/window；多图行 packing 折减 ~5%。

### 4.2 三档方案

| 档位 | S1 | S2 | S3 | 适用 |
|---|---:|---:|---:|---|
| **light** | **1%** | **1%** | **1%** | **首轮接入 + short run 推荐** |
| balanced | 2% | 2% | 3% | 扩数据到 ~4.4 万行后的目标档 |
| aggressive | 4% | 4% | 4% | 仅 balanced 明确正收益且无遗忘后 |

> **核心判断：先 light，不直接 balanced。** 当前试点只有 2,500 行；balanced 需要 6.0×/4.2×/8.4× 整池重复，
> 过拟合与重复效应会污染结论。light 已足以验证通道、长度、训练稳定性与初步 eval 信号。

### 4.3 首轮发射档位：light（整池重复 r=3/2/4）

> 数据来自 `mix_plan.json` 的 `recommended`（可机器读取；数值按试点实测 936 tok/行、基线 ipe 复算）

| 阶段 | 重复 r | 加入行数 | V6 tokens/epoch | 实际份额 | Δwindows | ipe 变化 | Δ iters/epoch | SAVE_STEPS |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| S1 | 3（2 epochs → 6 次曝光/行） | 7,500 | 7.02M | 1.029% | +823 | 2,631.8→2,657.5 | +0.98% | 886（原 512 为占位，不可沿用） |
| S2 | 2（2 epochs → 4 次） | 5,000 | 4.68M | 0.979% | +548 | 1,841.8→1,858.9 | +0.93% | 620（原 612） |
| S3 | 4（1 epoch → 4 次） | 10,000 | 9.36M | 0.967% | +1,097 | 3,730.2→3,764.5 | +0.92% | 1,255 |

- 实现：`tools/build_v6_stage_mixtures.py` 把 `dataset.parquet` 按阶段复制 r 份，副本 `sample_id` 加 `_rN` 后缀，保持行级键唯一；
- 已生成并校验（validator critical=0）：
  `output/datawheel_v6_pilot/stage_files/s1.parquet`（7,500 行）/
  `s2.parquet`（5,000）/ `s3.parquet`（10,000），
  并已打包进 `output/datawheel_v6_pilot/upload/{s1,s2,s3}.parquet`（图片共用 upload/images，117MB）；
- 成本：合计约 +100 iters ≈ +25 分钟（S1/S2 速度实测，S3 估算）；
- **这些值都以 smoke 打印的真实 ipe 为准重新计算**，不要直接手写死。

### 4.4 扩数据后 balanced 的行数需求（按试点 936 tok/row@512）

| 阶段 | 份额 | V6 tokens/epoch | rows@512/图 | rows@1024/图 | Δwindows | Δ iters/epoch |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 2% | 13.65M | ≈14,600 | ≈7,600 | +1,682 | +52.6（+2.0%） |
| S2 | 2% | 9.56M | ≈10,200 | ≈5,300 | +1,179 | +36.8（+2.0%） |
| S3 | 3% | 29.06M | ≈31,000 | ≈16,200 | +3,581 | +111.9（+3.0%） |

**关键杠杆：每行 QA 轮数。** 一条图片可承载 2~5 轮 QA / 多个任务视图；图片 token 占比高时，
提高每行轮数比增加图片数更省图片、编码与上传成本。正式生成按"目标 token ÷ 实测每行 token"反推行数。

### 4.5 分阶段 cate 目标配比（balanced 扩数据后使用）

| 阶段 | cate 行占比目标 | cate token 占比目标 |
|---|---|---|
| S1 感知 | apparel 35% / flag-hard 35% / reference_match 15% / misleading 15% | apparel 25.6% / flag-hard 25.5% / reference_match 37.8% / misleading 11.1% |
| S2 知识 | structured_caption 35% / text-compliance 25% / ip-brand 20% / anime-ip 20% | 41.6% / 22.8% / 17.7% / 17.8% |
| S3 hard+回放 | reference_match 35% / misleading 25% / quantity 20% / text-compliance 20% | 64.7% / 13.5% / 10.7% / 11.1% |

> 试点轮**不按该表加权**（200 行小 cate 在 1% 份额下需 15~30×/epoch 重复），按整池均匀重复；
> 该表用于扩数据后的自然配比。

### 4.6 balanced-v1 实际生成结果（2026-10-18）

已按上面的 union 配额生成并组装（仅 v2 train split）：

| 项 | 实际值 |
|---|---|
| union 唯一行 | **43,893**（单图 37,186 + 多图参考匹配 6,707） |
| S1 stage 文件 | **18,438 行**（apparel 6,453 / flag-hard 6,453 / reference 2,766 / misleading 2,766） |
| S2 stage 文件 | **15,656 行**（structured_caption 5,480 / text-compliance 3,914 / ip-brand 3,131 / anime-ip 3,131） |
| S3 stage 文件 | **19,164 行**（reference 6,707 / misleading 4,791 / quantity 3,833 / text-compliance 3,833） |
| 上传包 | `output/datawheel_v6_balanced/upload/`：s1/s2/s3.parquet + 7,982 张图片（314MB，前缀 `algo4/tpz/test/datawheel_v6_balanced`） |
| 本地校验 | `validate_swift_v6.py`：**critical=0**、exit 0；单行最长 4,210 tokens（1024/图保守口径） |

实测训练侧影响（512/图 token 估算，smoke 后以真实 ipe 复核）：

| 阶段 | V6 tokens/epoch | 实际 token 份额 | Δwindows | ipe | Δ iters/epoch | 建议 SAVE_STEPS |
|---|---:|---:|---:|---|---:|---:|
| S1 | 14.18M | **2.08%** | +1,662 | 2,631.8→2,683.7 | +1.97% | 895 |
| S2 | 9.99M | **2.09%** | +1,170 | 1,841.8→1,878.3 | +1.98% | 626 |
| S3 | 19.90M | **2.06%** | +2,331 | 3,730.2→3,803.0 | +1.95% | 1,268 |

**与目标的偏差与处理：**

- S1/S2 精准命中 ~2% 目标；**S3 只到 2.06%，未达 3%**。原因：参考匹配行实际约 1,899 tok/行（比 mix plan
  推导时的假设低），而 S3 的 token 预算高度依赖 reference 行。要到 3% 需要：
  (a) 把 reference 行扩到约 9.7k（+3k 行），或 (b) 每行候选参考图 2~3 张 → 3~4 张、并增加第二轮问答，
  把单行 token 提到 ~2,800。两者都留到下一轮（先看 smoke/eval 是否值得）。
- balanced-v1 相对试点（2,500 行）已是 **17.6× 行放大**；关键不是立刻拉满 3%，而是先跑通 smoke 与短程，
  用 eval 的 misleading / apparel / text-compliance / flag-hard 簇决定是否继续加量。
- 生成工具：`tools/build_v6_balanced_single.py`、`tools/build_v6_balanced_reference.py`、
  `tools/assemble_v6_balanced.py`（union → stage → validator → upload，一条命令可复现）。

---

## 5. 试点验证步骤（先试试看）

```bash
cd ~/Documents/Coding/datawheel

# 1) 本地生成与合并（无网络）
.venv/bin/python tools/build_v6_single_image.py
.venv/bin/python tools/build_v6_reference_rows.py --limit 650
.venv/bin/python tools/pack_v6_pilot.py --upload
.venv/bin/python tools/validate_swift_v6.py --dataset output/datawheel_v6_pilot/dataset.parquet

# 2) 远端上传
rsync -av output/datawheel_v6_pilot/upload/ \
    <remote>:/data/autovision-cfs/autovision-notebook/algo4/tpz/test/datawheel_v6/

# 3) smoke（10 iters）：S1 脚本 TRAIN_FILES 末尾追加 V6 parquet
TRAIN_FILES+=( "${TEST_DIR}/datawheel_v6/dataset.parquet" )
SMOKE=1 bash train_sft_qwen3_5_27b_megatron_lora_h200_vis_double.sh
```

验收门槛（全部满足才进入 light 正式 short run）：

- [x] `validate_swift_v6.py` critical=0，所有行 ≤8192（512 与 1024/图两口径均通过）
- [x] 图片路径/占位符/角色交替/`variant=6`/`enable_thinking=False` 全部通过
- [ ] smoke 10 iters 成功，日志出现 `loss_V6`
- [ ] `rank_pattern` / 视觉 LoRA / 冻结逻辑与基线一致；无 MaxLengthError、无静默丢行
- [ ] 上传后 remote 图片路径 100% 可解析（CWD=/data/autovision-cfs/autovision-notebook）
- [ ] 单行多图样本 `<image>` 数量与 images 数量一致（本地已验证）

---

## 6. 训练侧影响与重算

### 6.1 iters / SAVE_STEPS

- `新增 windows ≈ ceil(V6_tokens / (8114 × packing_factor))`，多图行 packing_factor≈0.95；
- `新增 iters/epoch ≈ 新增 windows / 32`；
- SAVE_STEPS 三套口径并存：历史 S1/S2 shell = `ipe/3`；S3 preflight 实际用 `ipe×epochs/6`；
  新 Python 调度器 = `round(rows/10.2/32/4)`（1/4 epoch）。**新增 V6 后必须显式重算并写入。**
- 本轮 light 示例（以 smoke 实测为准）：

| 阶段 | 原 total iters | 新增后 | 建议 SAVE_STEPS |
|---|---:|---:|---:|
| S1 | 5,263 | 5,315（+0.98%） | 886（/3 口径） |
| S2 | 3,683 | 3,718（+0.93%） | 620（/3） |
| S3 | 3,730 | 3,764（+0.92%） | 1,255（/3）；若按预检 /6 则 627 |

### 6.2 预期行数（light 推荐档）

- S1: 860,000 + 7,500 = **867,500**
- S2: 600,000 + 5,000 = **605,000**
- S3: 1,217,530 + 10,000 = **1,227,530**（若 S3 脚本默认含 V5 60k，则为 1,287,530，需先确认口径）

### 6.3 观测指标

| 指标 | 期望 |
|---|---|
| `loss_V6` | 平滑下降；若异常低/高，检查答案长度、图片 token 与难度 |
| eval_v3 5658 acc | 不低于基线（±0.2pp 内）；重点看 flag-hard / apparel / misleading / text-compliance 簇 |
| vision-strong 簇 | 不允许明显下降；若下降，先降 V6 份额/重复次数，再调 cate |
| 重复效应 | light 档 S1 每行 6 次曝光；若 loss_V6 极低但 eval 不涨，先扩唯一数据、不要加重复 |

---

## 7. 风险与硬规则

1. **S3 行数口径（高）**：任务口径 1,217,530（不含 V5）vs `vis_s3.sh` 默认 1,277,530（含 V5 60k）；
   windows/iters/SAVE_STEPS 相差约 5%，发射前必须二选一。
2. **重复/过拟合（高）**：light 档 S1/S2/S3 每行曝光 6/4/4 次；balanced-v1 已无重复（reference/misleading
   在 S1/S3 各出现一次，属顺序训练两次曝光）。S3 实测仅 2.06% 份额，若强行拉到 3% 需加参考行/图数，
   注意不要用简单重复去凑。
3. **S3 份额偏差（中）**：balanced-v1 S3=2.06% vs 目标 3%；优先看评测收益再决定是否扩 reference。
4. **视觉 token 估算误差（中高）**：512 token/img、packing 折减均为假设；远端真实值可能让
   Δwindows 偏移 ±20~30%，以 smoke 实测 ipe 为准。
5. **图片路径 base**：images 前缀 `algo4/tpz/test/datawheel_v6/…`，训练 CWD 必须是
   `/data/autovision-cfs/autovision-notebook`；首次编码会重新跑（几十分钟~小时级），需 NFS 缓存。
6. **cate_id 对账（高）**：当前 cate_id 是 Datawheel 自建能力簇；正式训练前需映射到 eval_v3 108 task 的 cate 表，
   否则分组指标无法对齐（reference_match 可能无直接 task，需用 Datawheel v2 held-out 自建指标）。
7. **V5 replay 隔离**：V6 不切验证集；in-run val 保持关闭（split_dataset_ratio=0）。
8. **行级键**：`(sample_id, cate_id, images)` 唯一；重复 key 会被预处理异常/去重，影响行数对账。
9. **不要修改 Mid_train 历史脚本**：接入只是 TRAIN_FILES 追加 V6 parquet；配方/sweep 冻结。

---

## 8. 本轮交付与下一轮

**本轮（goal round 2）已交付：**

- balanced union **43,893 行**（单图 37,186 + 多图参考 6,707），S1/S2/S3 stage 文件与 314MB 上传包；
- 本地 validator critical=0；S1/S2 实测 token 份额 2.08%/2.09%，S3 2.06%（3% 目标留待下一轮增强）；
- 生成工具链 `build_v6_balanced_single.py` / `build_v6_balanced_reference.py` / `assemble_v6_balanced.py`。

**上一轮（2026-10-18）已交付：**

- 方案主文档（本文件）；
- `tools/build_v6_single_image.py`：v2 → 单图多轮 QA / 结构化 JSON / 文本一致性行（1,850 行，确定性可复现）；
- `tools/build_v6_reference_rows.py`：v2 stage_d → 多图参考匹配行（650 行，商品图 + 2~3 候选参考图）；
- `tools/pack_v6_pilot.py`：parts 合并 / token 统计 / 远端上传包（2,500 行，3,225 图，117MB）；
- `tools/validate_swift_v6.py`：V6 数据契约与长度校验（critical=0；坏样本 9 类 critical 全覆盖）；
- `tools/build_v6_stage_mixtures.py`：按 light 档 r=3/2/4 生成 S1/S2/S3 三份 stage parquet 并打包上传；
- `output/datawheel_v6_pilot/`：dataset.parquet + upload/（dataset.parquet + s1/s2/s3.parquet + images）+ stage_files/ + validation/ + mix_analysis.md / mix_plan.json（本地）。

**下一轮：**

1. 远端 smoke（10 iters，建议先 s1.parquet）→ 确认 `loss_V6`、strict 不丢行、实际 ipe/视觉 token；
2. smoke 通过后跑短程 S1（如 1/10 epoch）评估 loss_V6 与 vision-strong/misleading/apparel 是否退化；
3. 若方向正确且需要 S3 3%：扩 reference 行到 ~9.7k 或提升单行参考图/问答密度；
4. 修正少量 cate 语义混杂（如 quantity 行里混入 identify 问题）并复核；
5. 补 eval_v3 108 task × cate 映射，建立 Datawheel v2 held-out 自建评测（reference_match/uncertain 无现成 task）。

---

## 附录：关键产物索引

| 产物 | 路径 |
|---|---|
| 试点训练集 | `output/datawheel_v6_pilot/dataset.parquet`（2,500 行） |
| 远端上传包 | `output/datawheel_v6_pilot/upload/`（dataset.parquet + images + README） |
| 契约校验 | `output/datawheel_v6_pilot/validation/`（critical=0） |
| 配比分析 | `output/datawheel_v6_pilot/mix_analysis.md` |
| 机器可读配比 | `output/datawheel_v6_pilot/mix_plan.json` |
| 单图 manifest | `output/datawheel_v6_pilot/parts/single_image_manifest.json` |
| 多图 manifest | `output/datawheel_v6_pilot/parts/reference_manifest.json` |
