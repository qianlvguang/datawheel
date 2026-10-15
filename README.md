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

## 📦 五大核心资产台账 (Asset Ledger)

| 资产类别 | 规模与覆盖 | 核心来源与规范 |
|---|---|---|
| **1. 官方国旗 / 国徽 / 地方行政旗帜** | **271** 款国旗 + **125** 款国徽/纹章 + **195** 款全球地方旗 | ISO 3166-1 权威矢量、Armorial of Sovereign States、美/加/德/西/日/澳/英/法/意/瑞各地方旗 |
| **2. 商业品牌与商标 Logo 库** | **3,466** 款官方商业品牌矢量 | Simple Icons 全量收录（运动潮流、数码科技、奢侈时尚、汽车零售） |
| **3. 动漫与角色 IP 库** | **19+** 款多姿态分体立绘 | PokéAPI（官方立绘/异色/像素/背影）、Kenney Monster Pack (CC0) |
| **4. 真实摄影级商品实拍底板库** | **9** 款高保真平铺原图 | 真实影棚微距实拍（黑/白/麻灰 T 恤、黑/灰卫衣、帆布包、马克杯） |
| **5. 外部独立评测基准集合** | **3** 大基准 (18 万+ 图像) | LogoDet-3K、QMUL-OpenLogo、ZACI-20 (物理隔离，专做泛化测试) |

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
│   └── ground_truth.py           # 0~1000 相对 BBox、密集 Caption 与多维视觉 QA
├── tools/                        # 资产拉取与台账总控工具
│   ├── asset_catalog_manager.py  # 资产巡检总控台账
│   ├── batch_asset_downloader.py # 批量轻量化资产拉取器
│   └── download_full_emblems_and_regional.py # 国徽与全球地方旗爬取器
├── output/                       # 合成产物输出
│   ├── images/                   # 摄影级合成实物图
│   └── datawheel_dataset.jsonl   # 训练与评测数据集 (ShareGPT/swift 兼容)
├── pipeline.py                   # 数据飞轮流水线主入口
└── README.md                     # 本项目技术总览
```

---

## 🚀 快速启动

### 1. 资产台账巡检
```bash
python3 tools/asset_catalog_manager.py
```

### 2. 启动数据飞轮合成流水线
```bash
python3 pipeline.py
```
产物将自动落盘至 `output/images/`，并同步导出符合 ShareGPT 格式的 `output/datawheel_dataset.jsonl`。
