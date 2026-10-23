# MAMMA 训练接口、保存恢复与有界框诊断（2026-10-15）

## 范围与结论

本批只用先前验收的六个观察槽，五个有 native oracle 框、一个物理域外无框。数据仍为 `integration_only`；更新产物另标 `infrastructure_smoke_only`，不可部署，也不构成效果微调。原始 282-key tensor state 严格加载（282/282，missing/unexpected/shape mismatch 均为零），模型四头结构未改变。框诊断全程使用原权重，先于任何参数更新；没有检测器、渲染、分割器或 TensorRT。

先前的候选来源、未知许可、KIT 演员独立性和预训练暴露限制继续适用；本批不划正式 train/val。同场景多 rig、邻帧或衍生衣发不拆作独立验证样本。

数据入口已改为显式 `mamma_observation_v1` manifest。新配置不在运行时读取审计 inventory；新增合格观察行只需遵循同一字段结构及资产白名单，不必修改 Python 的五槽列表。旧 pilot 配置继续读原 inventory，两个入口共用 `mamma_roi_pilot.py` 的 ROI 采样与身体投影。六个固定槽本次没有扩大；无框槽 `e02/rig_front/f0_cam4_p2` 保留 `outside_physical_domain_no_instance_box`，不可读取成训练样本。

## 数据字段和边界

manifest 行保存完整 ERP 尺寸、`raster.roi_xyxy`、stored ROI 尺寸、原框、人物/rig/episode、源整数 tick、身体帧、场景及相机矩阵引用。reader 返回图像、mask、512 有序 vertex ID、ERP/model/归一化 XY、完整 ERP 仿射及逆仿射、源采样矩形。`render_coverage_xyxy` 和 `point_in_render_coverage` 表示已存储的矩形 ROI 覆盖；`physical_valid_crop` 与 `physical_domain` 来自物理有效域 mask；`point_in_crop` 表示点在 384×512 crop 内；`source_sampling_supported` 表示实际插值和高斯预滤波源支持完整处于 ROI。支持越界直接拒绝。生产读取仅解码 ROI，没有完整 ERP 空画布。

旧 `coverage_crop` **实际是采样后的 physical valid mask**，现在保留为 `physical_valid_crop` 别名；旧 `crop_support` **实际是 point_in_crop**，现在保留为 `point_in_crop` 别名。旧 `roi_coverage` 是点落在存储 ROI 矩形内。新消费应使用具名字段，不能把它们互换。`coordinate_loss_valid` 由可靠几何、物理域、已存储 ROI、当前 crop 的交集决定；visibility/contact/floor-contact 均有独立 `value` 和 `loss_valid`。本批三类辅助标签为 -1/false。未来 manifest 的点标签必须明确 512 值、known；已知负例 0 与未知 -1 分开。资产白名单及有序 faces 核验仍实际执行，manifest 单方声明不能取得资格。

定向 CPU 回归：五个 native 样本的图像、人物 mask、仿射、坐标有效性逐元素一致；重算 ERP/model 浮点坐标最大差 6.82e-13 px。有效坐标点分别为 491、475、512、512、228。手算 loss 覆盖全有效、部分未知、全无效、NLL 两项同一 mask、未知辅助标签零梯度；全无效返回 `skip_optimizer`，不会做 weight decay 空更新。非法框与 ROI 尺寸拒绝。CPU 数值主检查 5.61 秒，含一次测试修正与调查仍远低于 20 分钟上限。

## Loss、参数组与恢复

`landmarks/lib/training/tasks.py` 提供训练/验证共用的任务有效性语义。坐标头输出第三值继续按旧 `log sigma`：平方残差先乘部位权重（本批固定 1）、再除 `2 exp(log_sigma)^2`、上限 25，另加 `2 log_sigma`；两个项均仅索引有效点，非法 GT NaN 不通过乘零进入计算。新归一化是**每样本有效权重和归一化，再平均有标签样本**，不同于原 `JointGNLLLoss` 对所有点的 mean。坐标项系数 100，visibility 有独立 masked BCE 开关；本批真实 visibility 全 unknown，实际贡献 0 且该头不入优化器。person-contact 与 floor-contact loss 关闭，头保留但参数冻结。所有任务无有效点时跳过实际更新并记原因。

AdamW 1e-5、weight decay 1e-4，无 scheduler/warmup。训练参数组：queries 393,216 标量、共享 decoder 47,268,864、landmarks MLP 1,183,491。ViT backbone、mask encoder、vis_prob、contact_prob、floor_contact_prob 冻结并排除优化器；每次前向前把冻结两个 encoder 置 eval，避免模型整体 `train()` 打开 drop-path。原 `configure_optimizers` 未纳入 mask encoder，本入口明确冻结它。共享 decoder 更新仍可能改变冻结辅助头的**预测**，只能保证其**参数**未更新。

从原 tensor state 严格加载的基础设施 smoke 采用固定 slot0 和 slot2，batch1、FP32、AMP/TF32 off、augmentation off、cuBLAS 确定性设置。完整 update1 checkpoint 保存模型、AdamW 状态、global update、CPU/CUDA/NumPy/Python RNG、固定采样位置与 epoch、框流状态、数据/配置/原 state 引用；`torch.load(weights_only=True)` 回读。实际 `optimizer.step` 日志只有 3 行：主分支 update1、主分支 update2、从 update1 恢复的 update2。恢复下一次 loss 两侧同为 **-781.2581787109375**，精确一致；冻结参数变化 0 项，允许组中 117 项发生有限变化。负 loss 源于 `2 log_sigma`，这里不是效果评价。checkpoint 为 931,404,407 字节，只在本任务目录，标记不可部署。

**恢复验收缺口：**第一次尝试因未设置 `CUBLAS_WORKSPACE_CONFIG` 在前向前停止，实际更新 0；定向重试做完主分支两次更新，但脚本在恢复分支错误地先释放 checkpoint 字典，第三次更新前停止。主分支 update2 的模型和 optimizer 内存快照随异常丢失。随后只从完好的 update1 checkpoint 续做剩余一次更新，达到总额 3；可核对下一次 loss，**不能直接核对主/恢复 update2 参数和 optimizer state 逐元素一致**。两份失败现场和步数日志均保留；不得为补这一项重跑任何更新。代码已修正释放顺序并抽出可复用 `training/recovery.py`，但未在额外更新上声称验收通过。

## 同图换框诊断

每父样本固定 native、水平中心 ±3% 原框宽、竖直中心 ±3% 原框高、同中心整体 0.9/1.1 尺度。35/35 框的完整采样支持均位于现有 ROI 内，无拒绝；先冻结了每父样本支持变体和共同 GT 集，五个共同点数分别为 **491、475、512、512、228**。slot0 native 新前向与旧原权重预测逐元素一致；五个 native 预测复用，另做 30 次扰动前向，总新前向 **31**。评价不以预测 visibility 或第三值筛点。

| 框变体 | 五样本共同集 ERP EPE 均值 (px) | 对 native ERP 预测漂移均值 (px) |
| --- | ---: | ---: |
| native | 8.513 | 0 |
| center x −3% | 8.743 | 1.965 |
| center x +3% | 8.203 | 1.740 |
| center y −3% | 8.124 | 1.576 |
| center y +3% | 8.848 | 1.819 |
| scale 0.9 | 8.143 | 2.113 |
| scale 1.1 | 8.868 | 2.917 |

35 框的共同集 EPE 简单均值 8.492 px；30 个扰动框的平均漂移 2.022 px，单框平均漂移最大 7.328 px。逐框的原有效集指标、共同集指标、框 ERP 坐标、预测及 GT/预测叠图见输出目录。共同集合以物理域、可靠几何和全部支持 crop 为准；此数值只刻画五个原权重样本的框敏感性，不是检测器框误差分布、泛化或效果提升。原样本不同视角不会被当作同一像素坐标系。

## 执行记录与下一步

工作目录：`/mnt/data1/yy/home/yy/Projects/mamma_retrain`；Python：`/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`，实际 dataset/tasks import 来自本副本。GPU 0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，运行前 18 MiB/24576 MiB；另一卡由他人使用。本批 GPU 数值运行与失败重试总计远低于 20 分钟，CPU 定向数值检查也低于 20 分钟。所有输入引用现有只读路径，结果仅写本副本 `outputs/experiments/mamma_training_readiness_box_20261015/`。没有安装依赖、参数更新之外的训练、提交或 SHA256。

查看 CPU 检查：`PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mamma-training-mpl /mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python scripts/check_mamma_training_readiness.py`。查看 smoke 计划：同一环境运行 `scripts/run_mamma_training_smoke.py`（无 `--execute`）；**当前 3 次更新额度已用尽，不得再次传 `--execute`**。框诊断与 smoke 的实际命令及逐步记录在输出目录；GPU 环境另设 `CUDA_VISIBLE_DEVICES=0` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

下一批最小动作：先从静态 checkpoint/日志审查恢复字段与脚本路径，待用户另行授权新的更新预算后才做分支状态逐元素复核；若要训练 visibility，数据侧至少需 512 点 known/可见/遮挡证据及物理域/裁剪映射；矩形框挑战扩大前需对应的完整 ROI 滤波支持。当前没有正式多任务训练、检测框分布验收或接触效果验收。
