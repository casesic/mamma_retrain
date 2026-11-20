# 跨仓库流水线契约

> **这份文件只有一份**，放在 `mamma_retrain`。另外三个仓库的 README 用绝对链接指向这里，不要各抄一份。
> 目的：说清四段之间**交接的文件**是什么、字段定义在哪、谁消费、边界在哪。
> 只写仓库文档里有证据的部分。没写明的连线单列在文末「仍是断点的地方」，不替作者补。

## 总览

交接单里画的是四段直线。按仓库文档和作者 2026-10-26 的双周报核对，真实情况是**两件事叠在一起**：

**（一）方法演进：一条路线替代另一条。** 不是串联，而是同一任务换了主方案：

```text
人体重建 → 投影            链路长，参数估计环节多
  → DenseKP               直接预测二维人体点（138 点）；两路独立预测可能不一致
  → DenseKP ＋ fitting    引入共同人体几何约束；此前有改善，多人遮挡仍薄弱
  → MAMMA                 实测整体更好 → 主方案转向 MAMMA（有序 512 点 + 可见性）
  → 微调 / fitting        合成数据指标改善；真实画面与原模型相近
```

**结论**：最终链路用 **MAMMA 的 512 点**，DenseKP 及其 fitting 路线**保留为历史对照**。合成数据工作最初是为 DenseKP 做的，后来才适配到 MAMMA——这解释了为什么生成器会反向引用 `densekp_retrain` 的产物目录。

**（二）仓库分工：一个数据源，两个训练消费者。**

```text
human_data_generator
  ├─ observations.json + RGB + 目标人物 mask + 有序 512 点
  │    ├──→ mamma_retrain    （微调 MAMMA 的 2D 点位模型，512 点）→ 真实合图
  │    └──→ densekp_retrain  （训练 DenseKP，138 点）→ 历史对照路线
  └─────→ Panorama-Stitch-Lab 在真实鱼眼/ERP 上做配对、变形与合图（主方案对应点取 MAMMA 512 点）
```

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

## 三个曾经断掉的地方，现在有答案了

这三条之前写在「待作者确认」，作者 2026-10-26 的双周报给了答案。**证据是仓库外的一份 PPT**（`20261026.pptx`，12 页），没有进 Git，所以下面对应到仓库文档的地方我尽量给仓库内的锚点；找不到锚点的会写明「仅 PPT」。

1. **512 点与 138 点不是上下游映射，而是两条竞争路线。**
   DenseKP 直接回归 138 个身体语义点（算子定义见 `densekp_retrain/docs/CONTRACTS.md` 的 `R[138,10475]`）；MAMMA 输出有序 512 点 + 可见性。两者都在解「跨相机同一人体点对应」，选型上 MAMMA 胜出，DenseKP 降为历史对照。所以仓库里**没有** 512→138 的索引表是正常的，不是缺失。

2. **真实合图用的是 MAMMA 的 512 点**，不是 DenseKP 的 138 点。
   最终链路「双机图像 → 检测/分割 → 原 MAMMA（512 点 + 可见性）→ 跨机配对筛点 → 变形合图」，其中原 MAMMA 是主方案、微调 MAMMA 是可选模型、MAMMA＋fitting 是可选分支。（仅 PPT；本仓 [DELIVERY](DELIVERY_20261025.md) 的合图对照同样是 512 点对，与之一致。）

3. **生成器反向指向 `densekp_retrain` 的产物目录，是历史依赖，不是设计。**
   `human_data_generator/config/resources.json` 里的 `calibration_dir` 指向 `densekp_retrain/outputs/.../prepared_scene/calibration`，因为数据生产最初服务 DenseKP。要彻底解耦需要把标定搬进生成器自己的资产目录——目前没有做。

## 还有一个没解开的：数字口径

2026-10-26 双周报里「微调 MAMMA」一栏写的是 **2.49 px / 可见 95.53% / 遮挡 91.18%**；这组数在本仓对应的是[自遮挡补负例试验](SELF_OCCLUSION_UNKNOWN_TRIAL_20261025.md)（该报告明确写「原数据包、默认标签规则和既有交付模型没有改动」）。而交付定稿用的是 812 步权重 + 0.85 阈值 → **91.33% / 94.76%**（见 [DELIVERY](DELIVERY_20261025.md)）。

作者 2026-11-20 确认：**对外统一用交付口径（0.85 阈值）**。双周报那组数字来自试验分支，不作为当前结论；两者不是同一份权重，不能混引。
