# MAMMA 输入 mask 派生框清理（2026-10-22）

## 结论与版本边界

当前正常训练数据入口为 `MammaMaskDerivedDataset`，配置为 `configs/retrain/mamma_mask_derived_reader_20261022.json`，裁剪 policy `mask_bbox_inclusive_v1`。它直接解码已有的二值目标人物 ROI mask，以 `cv2.boundingRect` 求局部外接矩形，再加 ROI offset 取得完整 ERP 框。右下端点是**最后一个前景像素**：`offset + min + size - 1`。这与只读部署比较路径 `riding-data/mamma_mask_roi.py:person_mask_roi` 返回的端点一致。部署先对预测概率 mask 做阈值和 resize；本路径使用已存储的二值 oracle 实例 mask，没有运行分割器，也没有伪装成预测 mask。

新框继续使用既有 1.2 扩张、384:512 宽高比适配、条件 Gaussian 预滤波、图像双线性/mask 最近邻采样和完整滤波支持检查。生产读取只解码 ROI；不建 5760×2880 空画布。`parent_box_xyxy` 是实际派生框，`manifest_box_xyxy` 仅保留历史来源核对；输出还带 `box_source`、`crop_policy_version`、完整 ERP 仿射及逆、ROI offset 与支持矩形。可靠坐标、物理域、已存储 ROI 覆盖和当前 crop 各自独立；坐标有效性不按 mask 值或未知 visibility 清零。支持不足直接拒绝，不缩框或补黑。

`MammaROIPilot` 和 `MammaManifestDataset` 仍是 `manifest_native_v1` 历史 reader，供 2026-10-13 接口验收、2026-10-15 原权重基线与七框诊断使用。旧配置和结果不变。正常 v2 reader 的 `get_with_box` 明确拒绝；v2 配置只要出现独立框增强、`box_override` 或旧 `future_box_augmentation` 字段就拒绝，正常配置不保留无效的 off/on 开关。`box_probe.py` 与对应运行脚本仅为历史诊断能力；它们仍调用旧 v1 reader，共享同一 ROI 采样数学，但不在正常训练输入链中。**随机框增强此前并未实现**；本次清理的是旧计划字段和可误用的公开换框入口，而非撤销已运行的训练增强。旋转、随机 mask 等一般增强仍为 off；裁剪所需的框、ERP 还原及七框历史证据均保留。

## 五个真实样本与端点差异

已读五个旧有框槽和一个 absent 槽；无框 `e02/rig_front/f0_cam4_p2` 继续保留 `outside_physical_domain_no_instance_box`，不造框。五个二值 mask 的外接框与旧 manifest 框左上端点相同，右、下端点都少 1；旧 manifest 在这些样本上使用末像素后一格的边界。五例全部换了 crop，故没有“框一致可沿用旧输入”的样本。

| 槽 | mask 派生完整 ERP 框，右下包含 | 相对旧框 Δxyxy | 新坐标有效点 | 新输入差异元素，图像 / mask | ERP 框公式还原最大误差 px |
| --- | --- | --- | ---: | ---: | ---: |
| e01 cam2 p1 | [4227,1285,4446,1878] | [0,0,-1,-1] | 491 | 312652 / 596 | 0.000035 |
| e01 cam4 p1 | [4181,1286,4425,1874] | [0,0,-1,-1] | 475 | 355614 / 661 | 0.000045 |
| e02 cam2 p2 | [1790,1330,1898,1628] | [0,0,-1,-1] | 512 | 513702 / 1235 | 0.000039 |
| e03 cam2 p2 | [3614,1153,3978,2020] | [0,0,-1,-1] | 512 | 586176 / 528 | 0.000063 |
| e03 cam4 p1 | [1437,1263,1633,1819] | [0,0,-1,-1] | 228 | 308975 / 696 | 0.000027 |

五例的 `sampling_rect_erp` 均有完整实际滤波/插值支持；所有坐标有效布尔值与旧 reader 相同，但图像、mask 与仿射均有数值差异。用已知 GT 的 `xy_target` 经部署框公式回完整 ERP，最大误差 0.000063 px；经保存的仿射逆变换最大误差小于 1e-12 px。前者的微差来自 float32 框/尺寸和运算路径，不调整框去贴旧数值。本批**没有新框下的模型预测**，历史原权重五样本点加权 8.960 px 只属于旧 `manifest_native_v1` 输入，不能直接称为新 policy 基线或精度改善。

小型 mask 用例验证 ROI offset、右下 `size-1`、mask 扩展后框随之变化和空 mask 拒绝。部署 `person_mask_roi` 对单像素前景会返回零宽高 `[x,y,x,y]`；本 reader 在 `native_plan` 前以 `degenerate_person_mask_box` 明确拒绝，避免无定义裁剪。该保护比部署准备函数严格；若日后需要完全覆盖退化输入，应由部署负责人单独决定其消费者行为，不能在训练侧静默制造框。既有五例均非退化。

## 活动任务开关与旧恢复合同

新增 `configs/retrain/mamma_training_active_20261022.json` 和只读 plan 入口。四头构造仍保留；当前 `loss_enabled` 为 coordinate=true、visibility=false、contact=false、floor_contact=false，`supervised_visibility=false`。`training/policy.py` 将活动 loss 和 AdamW 参数组绑定到该配置；visibility 头继续冻结并排除优化器。CPU 手算含 **1 个 known visibility 正例**，将 visibility logits 反转后，其 loss 仍为 0 且总 loss 不变，证明当前关闭不是依赖标签全 unknown。旧 `mamma_training_smoke_20261015.json` 的 visibility=true 只对应历史三次更新，原记录和配置均未改写。

`mamma_infrastructure_resume_v1` checkpoint 的 `box_stream` 仍作为 legacy、`augmentation=off` 的保存/恢复字段；新代码拒绝往该旧格式写入其它 augmentation 值，没有删除旧字段或改写 931 MB checkpoint。只做源码和配置静态检查，未加载完整模型/optimizer checkpoint。2026-10-15 三次真实更新额度已用尽，主分支 update2 的模型/optimizer 内存快照丢失造成的逐元素恢复比较缺口仍在；本批没有补验或新增更新。

## 验证、文件与后续决定

机器结果在 `outputs/acceptance/mamma_mask_derived_crop_cleanup_20261022/checks.json`、`static_checks.json` 与 `final_config_checks.json`；`progress.md` 记录里程碑。真实 Dataset 图像读取 **10 次**（旧/新各五次），另只读五张 mask 做端点预核；没有遍历 435 行，也没有读取 A3 新图像。CPU 定向回读内部计时 2.19 秒，连同 Python 初始化、两次只读 plan、预核和配置/静态检查的墙钟时间小于 2 分钟，低于 10 分钟上限；失败重试 0。模型前向 **0**、参数更新 **0**、GPU 调用 **0**、渲染 **0**。Python 为 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`；工作目录 `/mnt/data1/yy/home/yy/Projects/mamma_retrain`。没有安装、下载、提交或资产校验和。

本批修改：`landmarks/lib/datasets/mamma_roi_pilot.py`（v2 框来源及隔离），`landmarks/lib/training/policy.py`（活动任务入口），`tasks.py`（显式 AdamW betas），`recovery.py`（legacy off 守卫），`box_probe.py` 与历史脚本 docstring；新增两份 2026-10-22 配置、只读 plan、CPU/静态检查脚本；更新 `docs/A3_DATA_HANDOFF.md`，新增本报告及 acceptance 文件。旧带日期配置、基线、框诊断、失败记录和 checkpoint 均保留。

总协调仍需决定：何时以及按何证据启用 A3 新的 512 visibility 监督；未来若评估分割误差，应先冻结预测 mask/扰动、派生框和支持策略，并另授权计算。A3 本轮无需改 renderer、consumer、packing 或编辑器。当前样本仍是 `integration_only`，来源许可、KIT 演员独立性及同场景多 rig 不拆 train/val 的约束继续有效。
