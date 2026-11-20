# 跨仓库流水线契约

> **这份文件只有一份**，放在 `mamma_retrain`。另外三个仓库的 README 用绝对链接指向这里，不要各抄一份。
> 目的：说清四段之间**交接的文件**是什么、字段定义在哪、谁消费、边界在哪。
> 只写仓库文档里有证据的部分。没写明的连线单列在文末「仍是断点的地方」，不替作者补。

## 总览

交接单里的画法是直线：

```text
合成数据（human_data_generator）
  → DenseKP 重训（densekp_retrain）
  → MAMMA 微调与评价（mamma_retrain）
  → 真实合图（Panorama-Stitch-Lab）
```

按仓库文档核对下来，实际是**一个数据源、两个训练消费者**：

```text
human_data_generator
  ├─ observations.json + RGB + 目标人物 mask + 有序 512 点
  │    ├──→ mamma_retrain    （微调 MAMMA 的 2D 点位模型，512 点）
  │    └──→ densekp_retrain  （训练 DenseKP，138 个语义点）
  └─────→ Panorama-Stitch-Lab 读 DenseKP 的 138 点模型做真实合图与 staged-LM 拟合
```

证据：`human_data_generator/README.md` 写「供 MAMMA 训练使用」；`densekp_retrain/docs/README.md` 把「MAMMA 512 当前生成入口」指向 `human_data_generator`；`Panorama-Stitch-Lab` 的交付链说明读的是 DenseKP 138 点（[densekp_production_delivery_refactor](https://github.com/casesic/Panorama-Stitch-Lab/blob/ssh_main/docs/Architecture/formulate/densekp_production_delivery_refactor.md)）。

## 第 1 段：合成数据（human_data_generator）

| | |
|---|---|
| **输入** | 场景 JSON（`config/scene_example.json`、`config/batch_rules.json`）、人物/动作/衣物/发型资产、相机标定、Blender |
| **输出** | `outputs/runs/<任务>/rigs/<rig>/frame_XXX/<cam>/` 下 `roi.png`、`person_<id>_mask.png`、`mamma_512.npz`、`annotations.npz`、`raster_storage.json`；汇总文件 `observations.json` |
| **格式定义在哪** | `src/human_data_generator/data/mamma_export.py`（导出与一致性校验）、`docs/CODE_GUIDE.md`（一个样本的完整旅程） |
| **谁消费** | `mamma_retrain` 的 `MammaSavedGTDataset`；`densekp_retrain` 的 Dataset |
| **不含** | 检测框（由目标人物 mask 在训练端算出来）、接触标签 |

要点：一条 `observations` = 一个（帧, 相机, 人）；没有可见 mask 的人仍记为 `absent`；512 点的 XY 是**完整 ERP 坐标**，`roi.png` 只是局部图，位置在 `raster_storage.json` 里。

## 第 2 段：DenseKP 重训（densekp_retrain）

| | |
|---|---|
| **输入** | 上一段的数据包（清单 + RGB + mask + 512 点标签）与资产/场景 |
| **输出** | 138 点模型及其预测、评价指标（`outputs/` 不进 Git） |
| **格式定义在哪** | `docs/CONTRACTS.md`：138 点 = `R[138,10475] @ vertices[10475,3]`；ERP 经纬度与相机矩阵约定；可见性三态 |
| **谁消费** | `Panorama-Stitch-Lab` 的 DenseKP runtime（138 点 → `ERP_ROTATE` observation → C++ staged LM） |
| **边界** | 不做最终合图、不做部署流水线（README 已划） |

## 第 3 段：MAMMA 微调与评价（mamma_retrain，本仓）

| | |
|---|---|
| **输入** | 同一批 `mamma_bundle_v1` 数据包：`observations.json` + RGB + 目标人物 mask + 有序 512 点标签 |
| **输出** | 微调 checkpoint、可见性阈值研究、合成验证 7 场景 814 条指标、真实合图对照 |
| **格式定义在哪** | [MAMMA_CURRENT_ENTRY](MAMMA_CURRENT_ENTRY.md)（当前入口）、[CITY_MERGED_FULL_EPOCH](CITY_MERGED_FULL_EPOCH_20261025.md)（指标口径）、[DELIVERY](DELIVERY_20261025.md)（权重校验值与保存范围） |
| **谁消费** | 本仓自己的真实合图对照（`scripts/real_riding_stitch_compare.py` 读配对与合图模块） |
| **边界** | 不替换部署 engine；真实图片没有人工点位真值，判断是总体画面意见，不是逐帧准确率 |

## 第 4 段：真实合图（Panorama-Stitch-Lab）

| | |
|---|---|
| **输入** | 真实双相机鱼眼 → ERP；DenseKP 138 点；C++ staged-LM 拟合 |
| **输出** | 跨相机构图、单帧拟合性能记录 |
| **格式定义在哪** | [ADR-0005](https://github.com/casesic/Panorama-Stitch-Lab/blob/ssh_main/docs/adr/0005-densekp-production-delivery.md) 与 [densekp_production_delivery_refactor](https://github.com/casesic/Panorama-Stitch-Lab/blob/ssh_main/docs/Architecture/formulate/densekp_production_delivery_refactor.md) |
| **边界** | composite 阶段是效果验证；最终合成方向（dense flow）还未定 |

## 四段共用的硬约定

1. **坐标**：一切跨段交接都用完整 ERP 像素坐标；局部 ROI 的位置由 `raster_storage.json` 说明，不要把局部坐标当全局坐标。
2. **可见性三态**：`1` 可见、`0` 被遮挡、`-1` 无法确定；`-1` 不取消坐标监督。
3. **数据不进 Git**：`outputs/`、`data_process/`、`datasets/` 都在各仓的 `.gitignore` 里，任何「跑通」都要自备。
4. **权重不进 Git**：靠 SHA-256 校验值固定版本（见 [DELIVERY](DELIVERY_20261025.md)）。
5. **文档不写本机绝对路径**：四仓这一轮都在清，历史文档里的旧路径按各自仓库的待确认清单处理。

## 怎么确认这条链没断

最小的一条检查是：在 `human_data_generator` 里导出一次 `observations.json`，然后拿它喂给 `mamma_retrain` 的只读检查（不加载模型、不占 GPU）：

```bash
# human_data_generator 侧
python -m human_data_generator.run export --root outputs/runs/<任务名>

# mamma_retrain 侧（配置文件里指向该清单）
python scripts/mamma_retrain.py plan       --config configs/retrain/mamma_regular.json --split validation
python scripts/mamma_retrain.py check-data --config configs/retrain/mamma_regular.json --split validation --limit 2
```

字段缺失、点序不一致、包内文件对不上会在这两步直接报错——这比读文档更能判断链条有没有断。

## 仍是断点的地方（文档里没写明，待作者确认）

1. **512 点与 138 点之间怎么对应**：`mamma_retrain` 用 512 点模板，`densekp_retrain` 用 138 点算子（`R[138,10475]`）。两者都从同一个身体网格出发，但仓库里**没有一份文档写明 512 → 138 的索引表或映射规则**。
2. **真实合图到底用哪条对应**：`mamma_retrain` 的真实合图对照用的是 512 点对（共同成功 35 帧、点对 2474 → 3665），`Panorama-Stitch-Lab` 的 DenseKP runtime 用的是 138 点。两条是不是会汇合、以谁为准，文档里没有结论。
3. **资产与标定的唯一来源**：`human_data_generator` 的 `config/resources.json` 指向 `densekp_retrain/outputs/.../prepared_scene/calibration`。也就是生成器反过来依赖了重训仓的产物目录——这是有意为之还是历史遗留，需要作者说明。
