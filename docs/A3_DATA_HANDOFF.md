# A3 数据交接（2026-10-13）

## 2026-10-22 最新划分范围决定

用户明确不要求识别同一动作录制的不同片段。A1/A3 不新增此类来源追溯或自动合并分组，不把该能力当作正式训练前的缺口。现有 episode、相同 asset_id/asset_manifest 的交叉检查继续保留，已有来源信息不删除；本次不要求调整 split 代码。此决定覆盖旧讨论中对不同动作片段额外识别的要求，不扩大为取消全部划分检查。

## 2026-10-22 A1 正常直读接口与 A3 新导出

A1 将本项目新增代码收拢至 `mamma_ext/`，正常 `MammaSavedGTDataset` 直接按人物行读取现有 `mamma_512.npz` 的有序 512 XY 与 visibility；只在初始化核对模板、合格资产及人物/整数时间/相机/manifest 标签一致性，取样不再加载身体时序缓存或场景矩阵。RGB/mask 的 mask 派生框与 ROI 裁剪未变，接触任务仍关闭。旧重算 reader 保留为历史/审计路径，不是正常入口自动回退。

A3 本日新导出 `mamma_generation_refactor_a3_20261022/observations.json` 仍为 `mamma_observation_v1`，新增 `mask_derived_crop_support` 并改变 observation ID；其 XY/visibility 指向原 pilot NPZ。A1 的数值对照冻结原 `probe_manifest.json`，另对新导出做只读 plan（10 observed、2 absent）及单样本图像回读，均通过；新支持字段在存在时与实际 mask 派生框/采样矩形核对。A3 完整 31 帧生产仍待另批验收。本轮不要求 A3 改文件或补 GT。详见 [结构整理报告](MAMMA_TRAINING_REFACTOR_A1_20261022.md) 与 [当前入口](MAMMA_CURRENT_ENTRY.md)。

当前 visibility 汇总仍是 1137 正例、0 可信负例、5007 unknown，`integration_only`；训练配置 `execution_authorized=false`。正常训练/验证未来需按来源资产和 episode 分组；同场景多 rig、邻帧和人物不能拆到两侧。以下旧日期章节保留当时状态，本节为当前入口说明。

## 2026-10-22 A1 简化合同已落地

A1 在独立副本内沿用 `MammaMaskDerivedDataset`，新增指向 A3 现有 `probe_manifest.json` 的本地 mask 派生 reader 配置，以及坐标＋visibility 的只读训练配置。核心 GT 接口为有序 512 XY 与 visibility `1/0/-1`；旧 `value/known[/valid]/evidence_ref` 仍可读，未知不补零。坐标/visibility 有效性由 A1 现有几何、physical domain、ROI 和 crop 逻辑内部计算；无须 A3 另加 GT 字段或改生成器。A1 用旧样本和 A3 一例完成 2 次真实 CPU 回读，未运行模型或训练。详见 [简化合同报告](MAMMA_SIMPLE_TRAINING_CONTRACT_20261022.md)。

A3 当前标签汇总仍为 1137 已知正例、0 可信负例、5007 unknown，`integration_only`，代表性负例证据待补；新配置 `execution_authorized=false`。两个接触任务关闭，不要求对应 GT。以下带日期段落保留当时状态；其中“下一批落实接口”“当前活动 visibility 关闭”已由本节的新实现和新配置取代，旧 coordinate-only 配置保留历史语义。

## 2026-10-22 最新用户确认：简化训练接口

输入 RGB＋目标人物 mask，框由 mask 派生；核心 GT 仅为有序 512 点二维坐标和 visibility 1/0/-1。坐标与 uncertainty 联合学习，visibility 监督 0/1、跳过 -1；两个接触任务关闭。相机有效域、ROI、实际裁剪及几何检查复用 Dataset 内部逻辑，不追加生成端标注任务。visibility=-1 不能直接清除可靠坐标监督；旧 known/证据字段继续兼容，不删除已有产物。

A1 下一批基于已完成的 mask 派生框清理落实上述接口，准备坐标＋visibility 配置并做定向 CPU 验证。下文“当前活动 visibility 关闭”描述上一批状态；最新方案允许实现该任务的训练配置，但不授权运行模型或更新参数，不把现有仅有正例的小试升级为合格训练数据。A3 本批无需修改生成器。

## 2026-10-22 A1 实施结果（优先于以下历史计划）

A1 已在独立 MAMMA 副本新增 `mask_bbox_inclusive_v1` 正常 reader：直接从已存储的二值目标人物 ROI mask 用 `cv2.boundingRect` 取框，加 ROI offset 后以包含右下末像素的端点进入原裁剪数学。五个旧槽的 mask 框均比旧 manifest 框的右、下端点小 1 像素，故原 8.960 px 基线输入与新 policy 不同；未运行新模型基线。正常 reader 拒绝独立框覆盖，当前活动训练配置将 visibility/contact/floor-contact loss 全部关闭。历史七框诊断及旧 checkpoint 的 off 状态保留，但不再构成独立框增强的实施计划。本轮 A3 无需改数据或生成器。将来是否启用 A3 已交付的点级 visibility、或评估预测 mask 误差，需总协调另行冻结标签证据、输入 mask 政策和计算预算。

## 2026-10-22 总协调决定更新

用户已确定本项目当前范围不做人与人接触和地面接触：A3 不制作两类 GT、不安排接触标签小试；A1 不训练这两个任务。floor-contact 不再是待决定或默认后续阶段。为兼容原权重保留输出头，冻结参数、排除优化器、关闭 loss；现有字段保持 unknown/valid=false，不补零。共享 decoder 更新不保证冻结头输出不变。本次只记录决定，不要求删除网络层或改动部署。

当前训练不安排独立框扰动，不再要求框增强 off/on 对照；裁剪框应由输入人物 mask 按实际部署规则派生。已核实 `riding-data/mamma_mask_roi.py` 使用二值 mask 的 `cv2.boundingRect` 生成框。框的扩张、宽高比、ROI 滤波支持及坐标还原仍保留。后续 A1 实现时核对边界约定；本次仅更新文档，未修改代码或运行配置。

已有七框诊断与可选实现保留。预测 mask 或 mask 扰动及派生框的联合评价属于后续候选方案，尚未授权执行。完整决定见 [总体设计的 2026-10-22 更新](../../densekp_retrain/docs/training/MAMMA_DATA_TRAINING_DESIGN_20261015.md)。本节优先于旧交接中要求独立框增强的安排；下文带日期的旧状态作为历史记录保留。

本轮 A3 无需修改。A1 已在独立 MAMMA 副本中只读现有 ROI、身体、相机与时间元数据，五有框槽通过裁剪回读；一个物理域外无框槽保持原状态。A3 继续拥有编辑器、recipe、服务和当前公共生成适配。

未来**必需，仅在要训练 512 visibility 或启用框增强时**：

- 点级遮挡：在 live 深度和实例缓冲仍可用时，用冻结的 512 顶点 ID 与同一人物/相机变换记录每点 known、physical-domain、可见/被挡判定及证据来源；旧 compact 的 R138 visibility 不可扩成 512。
- 矩形框增强 ROI：给已审核框族保存实际框、`1.2×`/384:512 扩张及 Gaussian halo 后的完整采样矩形、ROI 支持或明确拒绝原因；不以补黑代表渲染。

可选诊断：遮挡者 ID/深度差、物理边缘距离、框族统计。上述字段应在公共生成端负责人明确分工后实施，不写进 A3 编辑器 recipe 的模型私有数组。现有资源与旧产物保持只读；A1 后续负责新 Dataset 回读和基线。

## 2026-10-15 训练侧 schema 确认

A1 已在自己的 MAMMA 副本中使用 `mamma_observation_v1` manifest；A3 本轮仍无需改动。`render_coverage_xyxy` 是已存储 ROI 范围，`physical_valid_crop` 是物理有效域的采样 mask，`point_in_crop` 是点落入 MAMMA 384×512 crop，`source_sampling_supported` 要覆盖插值及预滤波 halo。旧 `coverage_crop` 实为 physical mask，旧 `crop_support` 实为 point-in-crop。未来若交接 visibility，最少需要与冻结有序 512 ID 对齐的点级 value（1 可见、0 被遮挡、-1 unknown）、known 标志、来源证据和同一帧/人物/相机引用；已知负例不能从缺失标签推断。未来扩大矩形框族时，每框必须有完整滤波源支持或明确拒绝原因。现有五例三类辅助标签仍全部 unknown；本批没有要求 A3 补样本或改生成端。
