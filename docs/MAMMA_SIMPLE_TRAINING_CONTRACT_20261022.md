# MAMMA 简化输入、GT 与训练接口（2026-10-22）

## 当前合同

- 输入是 RGB 和已存储的目标人物二值实例 mask。`MammaMaskDerivedDataset` 直接读取 ROI，从 mask 的 `cv2.boundingRect` 求完整 ERP 框（右下为最后一个前景像素），再沿用 `mask_bbox_inclusive_v1` 的 1.2 扩张、384×512 宽高比、Gaussian 预滤波与完整源像素支持检查。正常入口拒绝独立换框。旧 manifest 框仅供来源对照。
- GT 是 MAMMA 有序 512 点二维坐标及 visibility 三态 `1/0/-1`。新标签可用 `point_labels.visibility` 的 512 值数组或仅含 `value` 的记录，`0/1` 推导 known；缺失为全 `-1`。旧 `value/known[/valid]/evidence_ref` 原样兼容：显式 known=false 或 valid=false 不产生监督，known=true 与 value=-1、valid=true 与 known=false 明确拒绝。旧 unknown 统一返回 `-1`。
- `coordinate_loss_valid` 仍由已资格化身体、几何有限、physical domain、render coverage 和当前 crop 内点共同决定，与 mask 前景像素或 visibility 值无关。`visibility_loss_valid` 是三态 known（及旧 valid）与同一可监督区域的交集；`-1` 不进入 BCE。ROI/滤波支持不足继续拒绝，不补黑或默认为全有效。身份、整数来源时间、相机/场景引用、完整 ERP 仿射及逆变换均由原 Dataset 保留。
- 坐标 XY 与网络原始第三值继续使用现有 log sigma NLL 联合学习；没有单独 uncertainty GT。本批未改 NLL 归一化、残差截断、loss 系数和 AdamW 超参数。visibility loss 开启时 `vis_prob` 同时进入参数组；关闭时两者同时关闭。ViT backbone、mask encoder、person-contact、floor-contact 头冻结；两个接触 loss 关闭且输入可不含接触 GT。四头模型结构保持原权重兼容。共享 decoder 将来若更新，冻结头的输出不保证不变。

## 实施与入口

在已有适配层新增 `parse_point_label`，集中解释三态和旧 known/valid；现有任务 loss 对缺失的辅助 GT 视为全 unknown，同时保留有标签但缺失有效性掩码的拒绝。`policy.py` 接受原 coordinate-only v1 和新的简化 v2，强制 visibility loss 开关与 `supervised_visibility`/参数组一致，两个接触任务不能开启。没有第二套 Dataset 或训练框架。

新 reader 配置 [mamma_a3_mask_reader_20261022.json](../configs/retrain/mamma_a3_mask_reader_20261022.json) 只读指向 A3 原 `probe_manifest.json`，不改 A3 的旧 native reader、manifest 或 NPZ。新 [mamma_training_simple_20261022.json](../configs/retrain/mamma_training_simple_20261022.json) 明确坐标＋visibility、augmentation off、`execution_authorized=false`。只读计划：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mamma_simple_mpl /home/yy/Repos/mamma/.venv/bin/python scripts/plan_mamma_training_active.py
```

计划显示 10 个 observed 槽与 2 个 absent 槽；`ready_count` 仅是 manifest observed 数，**不是全量图像回读或训练资格验收**。旧 [coordinate-only 配置](../configs/retrain/mamma_training_active_20261022.json) 和历史三次更新记录保持原语义；原 8.960 px 五例基线属于旧框输入，不是本 policy 的定位结果。

## 定向 CPU 验证

命令：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home/yy/Repos/mamma/.venv/bin/python scripts/check_mamma_simple_training_contract.py
```

结果见 [checks.json](../outputs/acceptance/mamma_simple_training_contract_20261022/checks.json) 和 [plan.json](../outputs/acceptance/mamma_simple_training_contract_20261022/plan.json)。累计真实 Dataset 图像读取 **2 次**（旧无 visibility 1 次、A3 正例 1 次），本定向数值脚本 CPU 计时 **0.82 秒**；计划仅读 manifest/模板，不读真实图像。旧样本 491 个有效坐标点、visibility 全 unknown；A3 `e03_rig_back_f000_cam2_p1` 有 512 个有效坐标点及 108 个有效 visibility 正例、0 个负例。两者 RGB `(3,512,384)`、mask `(1,512,384)`、mask 派生框、源采样支持、ERP 仿射往返检查通过；最大往返误差分别约 `9.1e-13`、`1.82e-12` px。正常入口拒绝 box override。

人工 CPU 张量验证：visibility `1` 和 `0` 均进入 BCE 并有梯度，`-1` 梯度为零；同一可靠坐标在 visibility `1/0/-1` 时坐标 loss 不变；域/crop 外标签由适配层有效性排除；全 unknown 时仅 visibility 跳过，所有启用任务全无有效点时标记跳过 optimizer 更新。新三态、旧 known=false、旧冲突、缺失接触 GT 均核对；小型替身模块确认 visibility on/off 对应 `vis_prob` 参数组，两个接触头和两类 encoder 冻结。验证没有构造 MammaNet。

本批 **0 模型前向、0 optimizer.step、0 GPU 调用、0 分割器、0 渲染**，未加载大权重或恢复 checkpoint。旧完整模型/optimizer 跨分支恢复一致性仍未补验。

## 标签资格与后续边界

A3 汇总为 **1137 个已知正例、0 个可信负例、5007 个 unknown**，`training_qualification=pending_representative_negative_evidence`，全部仍是 `integration_only`。配置和 plan 表达已确定的任务接口，但不授权实际训练；本批不造负例、不将正例小试认定为效果训练集。若今后需要 visibility 效果训练，最小数据缺口是可复核的代表性遮挡负例及其 512 点证据；无需为本合同另造接触 GT 或多套几何 GT。A3 当前无需修改生成器。

## 本批修改文件

- `landmarks/lib/datasets/mamma_roi_pilot.py`：三态/旧标签集中解析。
- `landmarks/lib/training/tasks.py`：缺失辅助 GT 的 unknown 兼容。
- `landmarks/lib/training/policy.py`：v1/v2 开关与参数组约束。
- `configs/retrain/mamma_a3_mask_reader_20261022.json`、`mamma_training_simple_20261022.json`：A3 只读输入与新活动 plan。
- `scripts/plan_mamma_training_active.py`、`check_mamma_simple_training_contract.py`：只读计划与定向 CPU 验证。
- `docs/A3_DATA_HANDOFF.md` 和本报告；独立 acceptance 目录含 progress、checks、plan。
