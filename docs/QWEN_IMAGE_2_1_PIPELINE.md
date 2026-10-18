# Qwen-Image 2.1 编辑管线设计（Datawheel）

> 目标：用 Qwen-Image 2.1 提升 Datawheel 商品图的真实感与场景多样性，同时**不破坏**实体名、图案、bbox、QA 的数学真值。
>
> 核心原则：**生成模型只做"底图/背景/光照/材质"，品牌图案继续走确定性合成。**

---

## 0. 先确认模型形态

不同部署方式决定脚本怎么接：

| 情况 | 说明 | 推荐用法 |
|---|---|---|
| Qwen-Image（文生图） | 只能 prompt → image | 生成**空白商品/背景/场景**，再合成图案 |
| Qwen-Image-Edit（编辑） | 支持 image + mask + prompt | 背景替换、补光、遮挡、换材质；**冻结图案区** |
| ComfyUI | 工作流 API | 需要 workflow JSON 或 API 示例，我写 ComfyUI 适配器 |
| **vLLM-Omni** | `vllm serve ... --omni`，OpenAI 图像生成/编辑 API | 直接可用 `--protocol openai`，见第 6 节 |
| **SGLang Diffusion** | OpenAI 兼容图像 API | 直接可用 `--protocol openai`，启动命令见 SGLang cookbook |
| Diffusers / 自建 FastAPI | 自定义 HTTP 接口 | 按本文件第 6 节的 JSON 协议接 |

请先确认：**模型是 T2I 还是 Edit、服务端口、一个 curl 示例、显存/并发限制**。

---

## 1. 推荐主链路（P0，最稳）

```
Qwen-Image 生成空白商品底图 / 背景
      ↓
质量门禁（无文字/水印/畸形手/异常 logo）
      ↓
现有 engine/compositor.py 确定性合成品牌图案
      ↓
自动生成 bbox / QA / caption（沿用现有流程）
      ↓
可选：Qwen-Image-Edit 只改背景/光照（mask 冻结商品与图案）
```

好处：

- 图案是矢量资产 + 程序化透视，**实体、颜色、bbox 全部可控**；
- Qwen 只提供"真实产品摄影感"（面料褶皱、棚拍光、场景），不会画出错误品牌；
- 分割隔离、评测集、QA 逻辑不需要重写。

**不要**让 Qwen 直接生成"印有某品牌 logo 的商品图"作为训练正样本——文字/logo 会幻觉，bbox 无法保证，标签会污染。

---

## 2. 四类用法与风险

| 优先级 | 用法 | 输入 | 输出 | 标签影响 | 风险 |
|---|---|---|---|---|---|
| P0 | 空白商品/背景生成 | prompt | 新底图 | 无（后续合成图案） | 手/文字畸形，需过滤 |
| P1 | 背景替换/补光/换场景 | 商品图 + 背景 mask | 编辑图 | 无（商品与图案区冻结） | mask 漏改到商品 |
| P2 | 轻微机位/透视变化 | 商品图 + 相机参数/ControlNet | 编辑图 | bbox 需 homography 重算 | 几何不一致 |
| P2 | 遮挡/手部/杂物 | 商品图 + 目标区 mask | 编辑图 | bbox 不变；可能变成 E 负样本 | 遮挡范围不可控 |
| P3 | 生成相似干扰图案 | prompt | 图案图 | 需人工/规则确认 | 可能生成真实品牌，污染标签 |

P1 的 mask 必须**排除**商品主体和印花区域；编辑后用像素差/SSIM 校验保护区域，差异超过阈值直接丢弃。

P2 如果确实要改机位：只能做**整体单应变换**，用已知 homography 重算 bbox；不要让模型自由重绘几何。当前 `augment_v2_rotation.py` 已经覆盖"轻微旋转 + mask 重算 bbox"，成本远低于生成模型。

---

## 3. Prompt 模板

### 3.1 空白商品底图

```text
professional e-commerce product photo of a blank {category}, front view,
{color} fabric with natural folds, no logo, no text, no graphic, no print,
clean white background, softbox studio lighting, sharp focus, high detail
```

negative prompt：

```text
logo, text, watermark, letters, brand mark, graphic, print, pattern,
distorted anatomy, extra fingers, mannequin face, low quality, blurry
```

### 3.2 背景替换（P1）

```text
replace only the background with {scene}; keep the product, its print and
its geometry completely unchanged; natural shadow and consistent lighting
```

### 3.3 补光/材质（P1）

```text
relight the product with soft cinematic studio light from the left;
preserve the printed graphic, its exact shape and position
```

> Edit 类 prompt 一定要写 "keep the product/print unchanged"；但真正防线是 mask + 编辑后像素校验。

---

## 4. 质量门禁

每条生成/编辑结果入库前至少检查：

1. **尺寸/解码**：与源图一致（编辑）或符合目标尺寸（生成）；
2. **保护区域不变**：商品主体/印花区相对源图的 MAE/SSIM 超阈值 → 拒绝；
3. **无新增文字/水印**：OCR（PaddleOCR/Tesseract）或简单纹理检测；空白商品图尤其要保证无字母；
4. **无畸形手/脸**：可先用简单规则/CLIP 提示词相似度，再人工抽检；
5. **类别一致性**：CLIP/embedding 与目标品类 prompt 的相似度；
6. **可复现**：记录 model version、prompt、negative prompt、seed、steps、mask、源图 hash、输出 hash；
7. **比例控制**：训练集里生成数据建议 ≤ 20~30%，并按类别监控，不要覆盖真实长尾。

---

## 5. 数据配比与 provenance

建议先做一个小规模实验，不要一次全量：

| 数据来源 | 占比 | 用途 |
|---|---:|---|
| 现有真实底板 + 合成图案 | 70~80% | 主力 |
| Qwen 空白商品 + 合成图案 | 10~20% | 补品类/姿态/场景 |
| Qwen 背景/光照编辑（冻结图案） | 5~10% | 域多样性 |
| Qwen 遮挡/杂物编辑 | 5% | E 负样本/证据不足 |

在 sidecar manifest 中记录 `source_type`、`qwen_model`、`prompt_id`、`seed`、`edit_mask`、`source_image`；不要直接改现有 parquet schema，避免训练侧不兼容。

---

## 6. 脚本接口约定

本仓库的 `tools/qwen_image_ops.py` 支持两种协议：

### 6.1 JSON 协议（推荐自建 FastAPI 适配器）

```http
POST {base_url}/generate
Content-Type: application/json

{"prompt": "...", "negative_prompt": "...", "width": 1024, "height": 1024,
 "steps": 30, "guidance": 4.0, "seed": 42}

→ {"images": [{"b64": "..."}]}      # 或 {"images": ["<base64>"]}
```

```http
POST {base_url}/edit
Content-Type: application/json

{"image_b64": "...", "mask_b64": "...", "prompt": "...", "steps": 30,
 "guidance": 4.0, "seed": 42}

→ {"images": [{"b64": "..."}]}
```

### 6.2 vLLM-Omni / SGLang（推荐）

Qwen-Image-2.1 目前由 **vLLM-Omni** 和 **SGLang Diffusion** 支持，注意不是普通 LLM 的 `vllm serve`：

```bash
# vLLM-Omni（每个实例一个模型）
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

启动后端点与 OpenAI Images API 兼容：

- `POST http://<server>:8091/v1/images/generations`
- `POST http://<server>:8091/v1/images/edits`（multipart）

生成请求示例：

```bash
curl -X POST http://<server>:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen-Image-2.1","prompt":"a blank white t-shirt, no logo, no text","size":"1024x1024","num_inference_steps":40,"guidance_scale":1.0,"seed":42}'
```

编辑请求关键字段（vLLM-Omni 扩展）：

| 字段 | 说明 |
|---|---|
| `image` | 输入图（multipart 文件或 URL） |
| `mask_image` | 掩码；白色区域重绘（是否接受文件/URL/字符串以实际服务为准） |
| `prompt` | 编辑指令 |
| `negative_prompt` | 负向提示 |
| `num_inference_steps` / `guidance_scale` / `seed` | 扩散参数 |
| `size` / `output_format` | 输出尺寸/格式 |

SGLang 使用 **SGLang Diffusion** 的 OpenAI API（cookbook：Qwen-Image-2.1），启动参数以其文档为准；客户端同样用 `--protocol openai`、`--base-url http://<server>:<port>`。

> 当前 `tools/qwen_image_ops.py` 的 `openai` 协议按 vLLM-Omni 字段构造（`num_inference_steps`、`guidance_scale`、`mask_image`）。如果服务端 mask 只接受 URL/base64 而不是 multipart 文件，把实际 curl 发我，我按真实接口调整。

如果服务器是 ComfyUI：给我一个工作流 JSON 或 `/prompt` 调用示例，我再加 `--protocol comfyui`。

---

## 7. 使用示例

准备 `plans/blank_products.jsonl`：

```json
{"id": "blank_tshirt_0001", "category": "T恤", "prompt": "professional e-commerce product photo of a blank white cotton T-shirt, front view, no logo, no text, clean white background, softbox lighting, fabric texture, high detail", "negative_prompt": "logo, text, watermark, print, pattern, hands, distorted", "width": 1024, "height": 1024, "seed": 42}
{"id": "blank_bag_0001", "category": "双肩包", "prompt": "professional e-commerce product photo of a blank navy canvas backpack, front view, no logo, no text, clean white background, studio lighting, high detail", "negative_prompt": "logo, text, watermark, print, pattern, distorted", "width": 1024, "height": 1024, "seed": 43}
```

```bash
# 1) 批量生成空白商品底图（vLLM-Omni / SGLang 的 OpenAI 接口）
.venv/bin/python tools/qwen_image_ops.py generate \
  --base-url http://<server>:8091 --protocol openai \
  --model Qwen/Qwen-Image-2.1 \
  --plan plans/blank_products.jsonl \
  --out-dir output/qwen_gen/blank_products

# 2) 生成结果质检（尺寸/解码/可选保护区域）
.venv/bin/python tools/qwen_image_ops.py verify \
  --manifest output/qwen_gen/blank_products/manifest.jsonl \
  --min-size 512

# 3) 背景/光照编辑（mask 白色=可编辑，black=保护）
.venv/bin/python tools/qwen_image_ops.py edit \
  --base-url http://<server>:8091 --protocol openai \
  --model Qwen/Qwen-Image-2.1 \
  --mask-field mask_image \
  --plan plans/bg_edits.jsonl \
  --out-dir output/qwen_edit/bg \
  --protect-bbox 0,0,1000,1000   # 示例：整图保护，按需换成商品区域
```

> `--protect-bbox` 用 `ymin,xmin,ymax,xmax`（0~1000）；编辑后在该区域做像素差校验，超阈值移动到 `rejected/`。

---

## 8. 本地 mock 测试

`tools/qwen_image_ops.py` 已在本地 mock HTTP 服务上验证：
- `generate`（json / openai 两种协议）→ 图片 + manifest；
- `edit`（json）→ 图片 + 保护区域 `protect_diff=0`；
- `verify` → 解码/尺寸检查通过。

真实服务只需替换 `--base-url` 与协议；如 ComfyUI 等非标准接口，需补适配器。

## 9. 许可提醒

Qwen-Image-2.1 使用 **Qwen Research License**（研究/评估用途；商用需单独授权）。如果 Datawheel 仅用于研究训练，一般可用；若要把生成资产用于商业产品或分发衍生模型，需要先确认许可条款，并按要求标注 "Built with Qwen"。

资料：
- [vLLM-Omni 图像生成 API](https://docs.vllm.com.cn/projects/vllm-omni/en/latest/serving/image_generation_api/)
- [vLLM-Omni 图像编辑 API](https://docs.vllm.com.cn/projects/vllm-omni/en/latest/serving/image_edit_api/)
- [SGLang Diffusion Qwen-Image-2.1](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1)
- [Qwen-Image-2.1 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1)

## 10. 下一步

1. 确认服务协议/模型类型；
2. 先用 20~50 条 pilot 生成空白商品，人工看质量；
3. 质量可接受后，接入现有 compositor + V6 之后的数据构建流程；
4. 训练前保持生成数据比例 ≤20~30%，并单独评测真实/生成分布差异。
