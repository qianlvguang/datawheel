# Datawheel (多模态视觉数据飞轮与工业级实物微调评测系统)

面向多模态大模型视觉对齐、细粒度图元定位 (Visual Grounding)、主体身份识别与商品级实物合成评测的**程序化数据飞轮系统**。

---

## 🎯 系统定位与设计哲学

传统视觉微调面临**真实样本采集昂贵、边界框人工标注误差大、合成服装过于虚假（矢量课设简笔画）**等瓶颈。
Datawheel 采取**素材供应链与资产台账先行**原则，结合**纯程序化数学透视合成**，实现 100% 绝对数学真理的零幻觉数据生成：

1. **绝对数学真理 (Zero Hallucination)**：坐标系统一采用以**左上角为原点 (0, 0)** 的归一化相对坐标系，纵横范围均为 **0～1000**，格式严格为 `[ymin, xmin, ymax, xmax]`。
2. **摄影级真实商品底板**：彻底摒弃低保真手绘矢量假衣服，全面接入真实单反相机实拍的白底商品平铺大图（纯棉 T 恤、连帽卫衣、帆布托特包、陶瓷马克杯），具备真实微距纤维细节与自然悬垂褶皱。
3. **海量权威矢量图元**：覆盖全球标准国旗、全量商业品牌、主权国家国徽与世界主要地方行政区旗。

---

## 📦 六大核心资产台账 (Asset Ledger)

| 资产类别 | 规模与覆盖 | 核心来源与规范 |
|---|---|---|
| **1. 官方国旗 / 国徽 / 地方行政旗帜** | **271** 款国旗 + **125** 款国徽/纹章 + **195** 款全球地方旗 | ISO 3166-1 权威矢量、Armorial of Sovereign States、美/加/德/西/日/澳/英/法/意/瑞各地方旗 |
| **2. 商业品牌与商标 Logo 库** | **3,466** 款官方商业品牌矢量 | Simple Icons 全量收录（运动潮流、数码科技、商业零售） |
| **3. 奢侈品牌设计语言与权威参考清单** | **16** 大全球顶奢全套设计家族 | LV/Hermès/Chanel/Gucci/Dior/LOEWE 等 16 品牌设计语言、专利图纸与在线免下载数据源 |
| **4. 动漫与角色 IP 库** | **18** 大核心原型 (四大宇宙全覆盖) | 龙珠/海贼/火影/鬼灭/EVA/高达/吉卜力/三丽鸥/美战/马力欧/原神/宝可梦，接入 Danbooru/Manga109 |
| **5. 真实摄影级商品实拍底板库** | **9** 款高保真平铺原图 | 真实影棚微距实拍（黑/白/麻灰 T 恤、黑/灰卫衣、帆布包、马克杯） |
| **6. 外部独立评测基准集合** | **3** 大基准 (18 万+ 图像) | LogoDet-3K、QMUL-OpenLogo、ZACI-20 (物理隔离，专做泛化测试) |

> 资产全景详见文档：[`assets/ASSET_CATALOG.md`](assets/ASSET_CATALOG.md)  
> 结构化 JSON 清单：[`assets/manifests/`](assets/manifests/)

---

## 🏗️ 目录结构

```
Coding/datawheel/
├── assets/                       # 核心多模态图元与底板资产库
│   ├── official_flags/           # 271 款 ISO 3166-1 权威矢量国旗
│   ├── official_emblems/         # 125 款全球独立国徽、纹章与航天/组织勋章
│   ├── official_regional/        # 195 款全球省州地方行政旗帜 (美/加/德/西/日/英等)
│   ├── brand_logos/              # 3,466 款商业品牌官方矢量 (Simple Icons)
│   ├── character_ips/            # 动漫角色多姿态精灵图 (PokéAPI)
│   ├── commodity_substrates/     # 真实单反实拍商品平铺底板 (T恤/卫衣/包/杯)
│   └── manifests/                # 5 大资产结构化 JSON Schema 清单
├── engine/                       # 几何、色彩与物理合成引擎
│   ├── palette.py                # 纹章标准色彩字典
│   ├── geometry.py               # 矢量多边形与极坐标绘制
│   ├── builder.py                # 底板排版与图元组装
│   └── modifiers.py              # 挖空、撕裂与异构混合扰动器
├── synthesizer/                  # 标注与语义合成器
│   ├── ground_truth.py           # 0~1000 相对 BBox、密集 Caption 与多维视觉 QA
│   └── reference_qa.py           # 跨图多模态参考检索与接地对话协议
├── tools/                        # 资产拉取与台账总控工具
│   ├── asset_catalog_manager.py  # 资产巡检总控台账
│   ├── batch_asset_downloader.py # 批量轻量化资产拉取器
│   └── download_full_emblems_and_regional.py # 国徽与全球地方旗爬取器
├── output/                       # 合成产物输出 (按维度结构化落盘)
│   ├── gallery/                  # 官方 SVG 画廊图像与 gallery_dataset.jsonl
│   ├── contrastive/              # 七态物理商品实物图、参考卡与 SFT 对话数据集
│   ├── density/                  # 浓度阶梯实物图、积木实体卡与阶梯数据集
│   ├── demo/                     # 经典 Demo 图像与 demo_dataset.jsonl
│   └── datawheel_master_dataset.jsonl # 汇总全域 60 条样本的总微调训练集
├── generate.py                   # 🌟 统一全量生成总控脚本 (整合四大维度与 CLI 参数)
├── pipeline.py                   # 数据飞轮流水线主入口
└── README.md                     # 本项目技术总览
```

---

## 🚀 快速启动

### 1. 资产台账巡检
```bash
python3 tools/asset_catalog_manager.py
```

### 2. 启动数据飞轮合成流水线 (支持统一总控与各维度细分)

#### 一键执行全量 4 大生成维度流水线并生成合并总数据集：
```bash
python3 generate.py
# 或使用经典流水线命令
python3 pipeline.py
```

#### 按需生成指定维度数据集：
```bash
# 官方 SVG 8 大认知任务画廊 (数星星、找不同、视觉加减法、多增少补、国徽文字OCR等 24 样本)
python3 generate.py --mode gallery

# 七态物理融合实物对比矩阵 (T恤、马克杯、牛皮纸包装 21 样本)
python3 generate.py --mode contrastive

# 原创积木实体与 5 梯度空间与物理做旧浓度阶梯实验 (5 样本)
python3 generate.py --mode density

# 快速冒烟验证 Demo (10 样本)
python3 generate.py --mode demo
```

产物将自动结构化落盘至 `output/` 对应子目录，生成实拍图、0~1000 相对坐标 BBox 质检可视化图，并导出符合 ShareGPT / ms-swift 规范的 JSONL 数据集。
