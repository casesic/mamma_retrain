# 实验报告索引

这个仓库在 2026-10-13 → 2026-10-25 之间，按天记录了 17 份实验报告。它们是**当时的实验记录**，不是教程：想知道怎么跑，先看 [README](../README.md)；想知道当前状态，先看 [MAMMA_CURRENT_ENTRY.md](MAMMA_CURRENT_ENTRY.md)。

这一页只回答三个问题：**哪份说什么、现在还算不算数、从哪里开始读。**

## 状态怎么标

| 状态 | 含义 |
|---|---|
| **结论仍有效** | 当前代码 / 配置 / 口径仍以它为准，后面的报告没有推翻它 |
| **已被覆盖** | 有更晚的报告明确取代了它的结论或做法，最后一列写出被谁取代 |
| **仅历史** | 当时的过程记录或非交付试验：既不构成当前结论，也不是结果出处 |

判定是从严的：只有能指出**更晚的一份报告明确取代**时才标「已被覆盖」，说不清的一律标「仅历史」，不硬凑。

## 17 份日期报告

| 日期 | 报告 | 一行结论 | 状态 | 备注 / 被谁取代 |
|---|---|---|---|---|
| 10-13 | [RETRAIN_WORKSPACE_SETUP](RETRAIN_WORKSPACE_SETUP_20261013.md) | 建独立 clone 与写权限边界，本批不训练不提交 | 仅历史 | 接手记录，无结论 |
| 10-13 | [MAMMA_DATA_ADAPTER_PILOT](MAMMA_DATA_ADAPTER_PILOT_20261013.md) | 前六个固定槽位的适配器回读通过，产物只作 `integration_only` | 仅历史 | 该读取器后来被 mask 派生框 + `MammaSavedGTDataset` 取代 |
| 10-15 | [MAMMA_ORIGINAL_BASELINE_PILOT](MAMMA_ORIGINAL_BASELINE_PILOT_20261015.md) | 原权重 282 个 key 严格加载通过；五个合成样本点加权 8.960 px | 仅历史 | 8.960 px 是**旧框输入**口径，不能与 814 条口径的 4.3954 px 混比；其 `tensor_state.pt` 仍在用 |
| 10-15 | [MAMMA_TRAINING_READINESS_BOX_DIAGNOSTIC](MAMMA_TRAINING_READINESS_BOX_DIAGNOSTIC_20261015.md) | 训练/恢复跑到 3 次更新；换框漂移均值约 2.02 px | 仅历史 | 入口与恢复合同被 10-22 的重构替换 |
| 10-22 | [MAMMA_TRAINING_REFACTOR_A1](MAMMA_TRAINING_REFACTOR_A1_20261022.md) | 实现收拢到 `mamma_ext/`，`scripts/mamma_retrain.py` 成为唯一 CLI | 结论仍有效 | 仅恢复 schema 版本已从当时的 v2 升到 v7 |
| 10-22 | [MAMMA_MASK_DERIVED_CROP_CLEANUP](MAMMA_MASK_DERIVED_CROP_CLEANUP_20261022.md) | 输入框改由目标 mask 派生，策略 `mask_bbox_inclusive_v1` | 结论仍有效 | — |
| 10-22 | [MAMMA_SIMPLE_TRAINING_CONTRACT](MAMMA_SIMPLE_TRAINING_CONTRACT_20261022.md) | 简化训练合同：输入 RGB + 存储的 mask，512 点三态 visibility | 结论仍有效 | 文中的三个 `configs/retrain/*_20261022.json` 后来合并/移除，链接会 404 |
| 10-24 | [CITY_FULL_FIRST_FINETUNE](CITY_FULL_FIRST_FINETUNE_20261024.md) | 16/4 划分 500 步：点均误差 2.76 → 1.83 px，遮挡召回 97.83% → 87.96% | 已被覆盖 | 被 [CITY_MERGED_FULL_EPOCH](CITY_MERGED_FULL_EPOCH_20261025.md) 的 814 条统一验证取代为主结果 |
| 10-24 | [VISIBILITY_CLASS_MEAN](VISIBILITY_CLASS_MEAN_20261024.md) | `class_mean` 相对普通 BCE 少误判 6 个遮挡点，变化很小 | 已被覆盖 | 损失口径沿用，但结论被后续全量一遍实验重做 |
| 10-24 | [REAL_RIDING_MANUAL_REVIEW](REAL_RIDING_MANUAL_REVIEW_20261024.md) | 319 个真实目标的点位人工评审页，原模型 0.5 vs 微调 0.85 | 结论仍有效 | 当前点位评审入口；真实图无真值 |
| 10-25 | [OCCLUSION_30PCT](OCCLUSION_30PCT_20261025.md) | 30% 遮挡配额：少误判 15 个遮挡点，可见召回降约 0.56 个百分点 | 已被覆盖 | 现行配置已改为不使用额外遮挡配额 |
| 10-25 | [CITY_MERGED_OCCLUSION_30PCT](CITY_MERGED_OCCLUSION_30PCT_20261025.md) | 合并 500 步：旧验证 2.76 → 1.84 px，新验证 6.33 → 4.40 px，遮挡召回下降 | 已被覆盖 | 主口径改为 814 条统一验证 |
| 10-25 | [CITY_MERGED_FULL_EPOCH](CITY_MERGED_FULL_EPOCH_20261025.md) | 全量一遍 812 次更新：4.3954 → 2.5242 px（阈值 0.5） | 结论仍有效 | **头条数字的出处**；0.85 是另一工作点，不是本文口径 |
| 10-25 | [REAL_RIDING_STITCH_THRESHOLD085](REAL_RIDING_STITCH_THRESHOLD085_20261025.md) | 40 对真实合图：成功变形 37 → 35 帧；共同 35 帧点对 2474 → 3665 | 结论仍有效 | 真实图无人工点对真值，只能看可用点多少 |
| 10-25 | [SELF_OCCLUSION_UNKNOWN_TRIAL](SELF_OCCLUSION_UNKNOWN_TRIAL_20261025.md) | 4cm 双证据补负例：遮挡召回 85.94% → 91.18% | 仅历史 | 试验分支：原数据包与默认标签策略都未改。**注意**：2026-10-26 双周报把这条的 95.53% / 91.18% 当作「微调 MAMMA」汇报过；对外口径是本文 0.5 阈值下的 98.42% / 85.94%，两组不是同一份权重 |
| 10-25 | [ORIGINAL_MAMMA_BODY_FITTING](ORIGINAL_MAMMA_BODY_FITTING_20261025.md) | 原权重后接 SMPL-X fitting，画面与直接合图基本一致 | 结论仍有效 | 建议保留原 MAMMA 直接合图为默认路径 |
| 10-25 | [DELIVERY](DELIVERY_20261025.md) | 交付收尾：812 步权重，未替换部署 engine | 结论仍有效 | 本轮交付汇总，含权重校验值；文中另记了 0.85 阈值那组数 |

## 常驻文档（不属于按日期的实验报告）

| 文档 | 什么时候看 |
|---|---|
| [MAMMA_CURRENT_ENTRY](MAMMA_CURRENT_ENTRY.md) | **当前状态第一入口**：现在用哪个权重、哪个阈值、哪些说法已过期 |
| [DELIVERY](DELIVERY_20261025.md) | 交付边界：哪些产物不在 Git 里、迁移时要一起带走什么 |
| [CODE_GUIDE](CODE_GUIDE.md) | 要改代码时从哪读进去（数据、损失、恢复三条线） |
| [INSTALL](INSTALL.md) | 环境、CUDA、权重路径 |
| [CONFIGS](CONFIGS.md) | 训练/推理配置怎么改 |
| [DATASETS](DATASETS.md) | 上游五个数据集族与下载脚本 |
| [YOUR-DATA](YOUR-DATA.md) | 换成自己的拍摄数据要怎么摆 |
| [PIPELINE_CONTRACT](PIPELINE_CONTRACT.md) | 四个仓库怎么交接：每段的输入/输出/格式定义/消费者，以及仍是断点的地方 |
| [A3_DATA_HANDOFF](A3_DATA_HANDOFF.md) | 数据包格式（`mamma_bundle_v1`）与划分范围的决定 |
| [steps](steps.md) | 上游五步流水线（`ma_cap → … → ma_vis`）的代码位置 |

## 十分钟读法

1. [README](../README.md) 顶部的 fork 说明 → 知道这是 fork、我加了什么、结果多少。
2. [MAMMA_CURRENT_ENTRY](MAMMA_CURRENT_ENTRY.md) → 知道现在用哪个权重/阈值。
3. [CITY_MERGED_FULL_EPOCH](CITY_MERGED_FULL_EPOCH_20261025.md) 的「验证结果」一节 → 知道数字怎么来的、代价是什么。
4. [DELIVERY](DELIVERY_20261025.md) 的「保存范围」一节 → 知道哪些东西不在 Git 里、不能声称什么。
5. 想改代码再进 [CODE_GUIDE](CODE_GUIDE.md)。

## 两个读报告时容易踩的坑

**不同报告的数字不能直接比。** 训练数据（16 个场景 / 28 个场景）、验证集（435 条 / 379 条 / 814 条）、阈值（0.5 / 0.85）都不一样。典型例子：8.960 px（10-15，旧框输入、五个样本）和 4.3954 px（10-25，814 条统一验证）不是同一个口径，放一起比较没有意义。

**报告里的很多链接在本机以外点不开。** 指向 `../outputs/...` 的是本机实验产物（被 `.gitignore` 排除，本来就不进 Git）；指向 `../../human_data_generator/` 和 `../../densekp_retrain/` 的是**兄弟仓库**，在这个仓库里点不开，需要去各自仓库看。这些链接保留原样是为了可追溯，没有被删改。
