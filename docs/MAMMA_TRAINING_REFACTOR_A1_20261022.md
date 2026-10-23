# MAMMA 独立训练扩展与正常数据入口（2026-10-22）

## 结果与依赖方向

新增实现归入仓库根目录 `mamma_ext/`：`data.py` 是唯一共享 ROI 采样与新保存 GT reader，`upstream.py` 只装载当前克隆中固定的 crop 数学函数；`model.py` 放严格 tensor state 加载及 ERP 还原/指标；`tasks.py` 放原已验收的 masked NLL/BCE 和参数组；`policy.py` 绑定数据与任务开关；`recovery.py` 保存旧合同和新恢复字段；`split.py` 按来源/episode 防交叉；`runner.py` 组织只读 plan、数据检查、未来显式训练与评价。`scripts/mamma_retrain.py` 是推荐薄 CLI。扩展包只在明确模型执行路径才导入上游模型，不让上游源码反向依赖扩展。

旧 import 映射：`lib.datasets.mamma_roi_pilot` → `mamma_ext.data`；`lib.training.tasks/policy/recovery` → 同名扩展模块；`lib.baseline.mamma_original` → `mamma_ext.model`。这些旧文件已变为薄兼容包装，功能不再双份演化。历史 `lib.training.box_probe`、固定日期 baseline/smoke 脚本及所有既有输出留原位。上游跟踪文件本批未新增修改；`vitdet_dataset.py` 在开始前已有部署裁剪 diff，本批保留原状。没有改 `landmarks/train.py`、上游模型、包初始化或原仓库。

## 正常 Dataset 与 A3 合同

正常配置 [mamma_regular.json](../configs/retrain/mamma_regular.json) → [mamma_saved_gt_reader.json](../configs/retrain/mamma_saved_gt_reader.json) → A3 已有具名 pilot `probe_manifest.json`（`mamma_observation_v1`，10 observed、2 absent）。每个 observed 人物可用 episode、rig、frame、camera、person_id 定位。absent 保留原因，不造 mask/框。`MammaSavedGTDataset` 在初始化核对资产白名单、来源引用是否存在、有序 faces/512 顶点 ID、NPZ/annotations/manifest 的人物、资产帧、整数源时间、相机、ERP 尺寸、标签 policy 及 visibility 值/known 一致性；不对 `:array[index]` 字符串做 eval。NPZ 是正常 XY/visibility 权威来源，manifest 标签若冲突直接拒绝。

取样时直接从 `mamma_512.npz` 的对应人物行读 float32 `xy_full_erp`、`geometry_defined`、`physical_domain`、`render_coverage`、visibility；**不加载完整身体时序缓存、scene_matrices 或重投影**。RGB 与目标人物二值 mask 继续只从实际 ROI 读取；输入框由 mask 的包含右下端点求得，沿用 `mask_bbox_inclusive_v1` 的 1.2 扩张、384×512 宽高比、预滤波 halo 与插值支持检查。不构造完整 ERP 空画布、不提供独立 box override。新导出清单若带 `mask_derived_crop_support`，还要与实际 mask 派生框及采样矩形一致；旧 pilot 没有该字段时由 reader 自行检查。

坐标有效性由保存几何有效且坐标有限、物理 mask 与保存 domain、存储 ROI 与保存 coverage、实际 crop 的交集计算；保存与内部域判断不一致单列，采取保守交集，不用 mask 前景或 visibility 状态筛可靠坐标。三态 `1/0/-1` 和旧 known/valid 语义继续沿用，缺接触 GT 不阻塞。身份、源时间、相机/场景及 XY 引用、仿射/逆变换、ROI offset 均保留在输出。旧 `MammaMaskDerivedDataset` 仍可为没有 512 文件的历史样本重算几何，但 v3 正常入口缺件会拒绝，不自动切回旧 reader。

A3 结构整理文档在本批执行过程中交付；[新导出清单](/mnt/data1/yy/home/yy/Projects/densekp_retrain/outputs/experiments/mamma_generation_refactor_a3_20261022/observations.json) 仍为 `mamma_observation_v1`，新 observation ID 较长、加 `mask_derived_crop_support`，GT 文件仍引用同一 pilot NPZ。A1 的数值对照冻结旧具名 pilot 清单；对新导出只读完成 10 observed/2 absent 的 plan 与 1 项图像检查，均通过。A3 的完整 31 帧生产尚无 RGB 验收，本批没有把准备计划当数据。未知资产仍需单独登记，不要求 A3 为本轮增补字段。

## 入口、训练及评价边界

[当前入口说明](MAMMA_CURRENT_ENTRY.md) 给出命令。无参数帮助、`plan`、`check-data` 不构造模型或读取权重。`train` 缺显式更新预算、验证清单、输出目录或 `--execute` 时拒绝；当前配置 `execution_authorized=false`，即使提供完整参数也在模型加载前拒绝。未来获单独授权的配置还需 visibility 标签资格为 `qualified`，不能把当前全正例小试当效果训练集。输出只能在本仓库 `outputs/` 新目录。

未来训练组织代码使用原 282-key tensor state 的严格加载、坐标 log sigma NLL＋visibility BCE、AdamW 现有超参数及 query/shared decoder/landmarks/有监督时 vis_prob 参数组；冻结 ViT、mask encoder 和两个接触头，接触 loss 始终关闭。全无有效任务时不做 optimizer.step。训练/验证清单需按 episode、source asset/manifest 不交叉；同场景多 rig、邻帧与不同人物不能随机拆行。共享 decoder 更新后两个冻结接触头的**预测**不保证不变。

新 checkpoint `mamma_regular_resume_v2` 记录模型、optimizer、实际 update、CPU/CUDA/NumPy/Python RNG、采样 epoch/位置以及配置、train/validation manifest、模型配置、原 tensor state、crop/GT policy 和 augmentation=off 引用；旧 `mamma_infrastructure_resume_v1` 和 legacy `box_stream=off` 读写合同保留。当前仅做[小型合成状态 `weights_only` 序列化检查](../outputs/acceptance/mamma_training_refactor_a1_20261022/recovery_fields.json)；**真实模型/optimizer 跨分支恢复仍未验收**，旧三次 optimizer 更新预算已用尽。

未来评价组织代码沿部署公式还原完整 ERP XY，并与仿射逆变换核对；坐标主指标按 `coordinate_loss_valid` 的点加权及有标签样本等权均值，不按预测 visibility 或第三值筛点。预测非有限值保留在分母并报告。visibility 仅在真实 0/1 两类都存在且 logits 有限时报告分类指标；单一类别只报数量和不可评价状态。不混用 DenseKP 指标或合图结论。训练/评价执行路径在本批**未运行**，不能把组织代码或 plan 成功当实际训练验收。

## 有界 CPU 验收

机器记录：[checks.json](../outputs/acceptance/mamma_training_refactor_a1_20261022/checks.json)、[pilot plan](../outputs/acceptance/mamma_training_refactor_a1_20261022/plan_after_provenance_check.json)、[A3 新导出 plan](../outputs/acceptance/mamma_training_refactor_a1_20261022/a3_export_plan.json) 与 [失败记录](../outputs/acceptance/mamma_training_refactor_a1_20261022/failure.json)。两个 target 为 e03/rig_back/f000/cam2 的 p1、p2，各用旧重算与新直读一次。图像、mask、框、ERP 仿射及逆变换逐元素相同；float32 保存 XY 对 float64 旧投影的绝对误差如下：

| target | 最大差 ERP px | 平均差 ERP px | 坐标有效点旧/新 | 坐标/visibility 有效掩码变化 |
| --- | ---: | ---: | ---: | ---: |
| p1 | 0.000244 | 0.0000771 | 512/512 | 0/0 |
| p2 | 0.000122 | 0.0000454 | 512/512 | 0/0 |

新取样的受控 `np.load` 代理确认身体顶点、faces、scene matrices **热路径读取均为 0**；faces 只在初始化核验一次。小 fixture 拒绝错有序 ID、人物、整数时间、相机引用、manifest/NPZ visibility 冲突、crop 支持不足及独立换框；absent reason 保留。旧导入是同一扩展实现别名。CLI 的真实拒绝与 mock 分发、同 episode 跨 train/val 拒绝、单类 visibility 不报准确率、新恢复字段合成序列化均通过；未构造小模型，也未更新替身参数。

首次检查在模拟授权配置处发现 v2 policy 把 `execution_authorized=false` 写死，已修正为 v2 接受显式布尔值、v1 继续只读；失败记录保留。检查脚本成功轮读图像 4 次、失败轮已读 4 次，再加 pilot 与 A3 新导出各一次 `check-data`，本批真实 Dataset 图像读取**累计 10 次**（上限 12）；成功轮内部 CPU 检查 4.496 秒，含失败重试、两次单项回读及小张量补核的累计数值工作墙钟低于 2 分钟（20 分钟预算内）。模型构造/前向 **0/0**、optimizer.step **0**、GPU **0**、分割器 **0**、渲染 **0**。未读取大模型或 optimizer checkpoint，未改共享环境、下载、提交或计算 SHA256。

## 限制与修改范围

当前 A3 visibility 为 **1137 已知正例、0 可信负例、5007 unknown**，`integration_only`，标签 policy 仍待代表性遮挡负例及官方定义核证。新 reader 的两人对照不等于 10 人全部图像/边界验收，更不等于新输入原权重定位基线；旧 8.960 ERP px 属于 `manifest_native_v1`。A3 新导出的完整 31 帧渲染/打包也未被本批验收。若将来添加正式 train/val，必须有来源与 episode 层级清单和合格 visibility 证据。

本批修改：新增 `mamma_ext/{__init__,upstream,data,model,tasks,policy,recovery,split,runner}.py`；新增 `configs/retrain/{mamma_saved_gt_reader,mamma_regular}.json`；旧 `landmarks/lib/datasets/mamma_roi_pilot.py`、`landmarks/lib/training/{tasks,policy,recovery}.py`、`landmarks/lib/baseline/mamma_original.py` 改为兼容包装；新增 `scripts/{mamma_retrain,check_mamma_training_refactor}.py`，旧 `scripts/plan_mamma_training_active.py` 改为薄包装；新增本报告及当前入口说明，更新 `docs/A3_DATA_HANDOFF.md`，新增独立 acceptance 目录。其它带日期产物、预测、旧 checkpoint、失败记录均保留。
