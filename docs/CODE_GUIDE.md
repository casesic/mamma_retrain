# 从哪里读懂 MAMMA 微调代码

如果只是想检查数据或了解当前限制，先看[当前使用说明](MAMMA_CURRENT_ENTRY.md)。这份指南写给要修改我们自写代码的人；克隆的 MAMMA 模型源码在 `landmarks/`，本轮微调入口不会改动它。

## 从一条命令跟进去

入口是 `scripts/mamma_retrain.py`，它把命令交给 `mamma_ext/runner.py`。从 `runner.main` 往下读，会看到两条清楚的路径：

1. `plan`、`check-data`：先由 `mamma_ext/policy.py` 读取训练配置，再由 `mamma_ext/data.py` 打开 `mamma_bundle_v1` 清单。`plan` 只检查文件与标签合同；`check-data` 再解码少量 RGB 和 mask。两者不会创建 MAMMANet、加载权重或运行训练。
2. `train`、`evaluate`：`runner.py` 先检查显式执行条件，然后才进入模型路径。`mamma_ext/model.py` 负责原权重严格加载、预测坐标还原和评价指标；`mamma_ext/tasks.py` 负责坐标/uncertainty 与 visibility 损失、参数冻结和优化器分组；`mamma_ext/recovery.py` 保存或读取更新边界。训练/验证来源交叉检查在 `mamma_ext/split.py`。

不必先把整个 `landmarks/` 包读完。普通数据检查只会通过 `mamma_ext/upstream.py` 取用两项已有裁剪数学函数；模型构造留在明确的执行路径中。

`scripts/export_trusted_mamma_state.py` 是单独的权重准备工具：它把既有、固定来源的原 checkpoint 导出成当前配置引用的纯 tensor 状态。日常 `plan`、`check-data` 不调用它，也不读取原 checkpoint；这不是另一套训练入口。

## Dataset 为什么分成几步

`data.py` 初始化 Dataset 时，先检查 reader 配置和清单格式，再逐观察记录核对 episode、人物、相机、整数时间、模板点序、ROI 和标签文件。它只依赖数据包内的相对路径，不回读生成器的身体缓存、动作文件或场景矩阵。只有状态为 `observed` 的记录可以取样；`absent` 仍保留在清单中供统计。

取一个样本时，它才读取 RGB 和目标人物 mask，从 mask 求**包含最后一个前景像素**的框，按固定比例扩张、适配模型输入，并检查高斯滤波和采样所需区域是否完整落在已保存的 ROI 内。之后读取有序 512 点，转换到 384×512 模型裁剪坐标。标签中的 XY 始终也保留完整 ERP 坐标；`visibility=-1` 不会抹掉本来有效的坐标监督。

修改 Dataset 时，先看 `MammaSavedGTDataset`、`mask_box_inclusive`、`native_plan` 和 `_targets`。尤其要保留 mask 派生框右下端点、点序、身份时间、坐标系与采样支持检查。`tests/test_current_retrain.py` 用小夹具检查拒绝行为；历史真实包只读用于少量样本对照。

## 修改监督或恢复时

训练配置在 `configs/retrain/mamma_regular.json`。`loss_enabled.visibility` 同时决定 visibility 损失是否启用、visibility 头是否加入优化器。接触头虽然留在模型结构里，但没有接触损失，且不加入优化器。损失公式与空监督跳过更新由 `tasks.py` 负责；改这里应先用小张量对照数值、unknown 标签和空样本。模型头列表表达原权重的结构，不代表这些任务正在训练。

更新边界在 `recovery.py`：正常文件是 `mamma_regular_resume_v4`。它校验当前配置与数据引用，严格恢复模型和优化器，并恢复 Python、NumPy、PyTorch RNG 以及采样位置。旧 smoke 使用的带 `box_stream` 的 v1 恢复链已经移除。更改 checkpoint 内容或含义时必须换 schema，不能让旧文件在相同版本下被误读。这里的假模块测试证明保存/读取逻辑，不等于真实 MAMMA 训练恢复验证。

如果只想改变数据清单路径，改 reader 配置或在命令中传 `--manifest`；如果只想改变训练任务开关，改训练配置。两个项目通过数据包交接，不互相导入实现。当前 `execution_authorized=false` 和 visibility 标签资格限制不能因整理代码而绕开。
