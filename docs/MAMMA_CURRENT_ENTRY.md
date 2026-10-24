# MAMMA 微调：从数据检查开始

这个项目读取相邻的 `human_data_generator` 项目产出的 `mamma_bundle_v1` 数据包。训练输入是一张 RGB 图和这次目标人物的 mask；裁剪框由该 mask 计算，不单独提供框或做框增强。标签是固定顺序的 512 个完整 ERP 二维点及每点可见性：`1` 可见、`0` 被遮挡、`-1` 未知。每个人都有自己的观察记录；没有可用 mask 的记录留在清单中，标为 `absent`。

目前只有两份正常配置：[训练配置](../configs/retrain/mamma_regular.json)控制任务和执行条件，[reader 配置](../configs/retrain/mamma_saved_gt_reader.json)指出数据清单。配置中的项目内路径相对配置文件解析。reader 默认清单指向未来的 `key_batch_001`，该批数据尚未生成；检查已存在的数据时，请用 `--manifest` 指定实际清单的**绝对路径**。

## 先看数据是否能读

在项目目录运行：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
MANIFEST=/mnt/data1/yy/home/yy/Projects/human_data_generator/outputs/runs/first_current_render_a1_20261023_pair_walk_densekp_env/observations.json

env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 \
  "$MAMMA_PY" scripts/mamma_retrain.py plan --manifest "$MANIFEST"

env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 \
  "$MAMMA_PY" scripts/mamma_retrain.py check-data --manifest "$MANIFEST" --limit 2
```

`plan` 只核对清单、标签文件、身份时间、点序、分组和配置，不解码图片，也不加载模型。`check-data` 再读指定数量的 observed 样本，报告图片和 mask 形状、可监督坐标点数与可监督 visibility 点数；它仍不构造模型。上面的短真实渲染包有 8 个观察槽，其中 4 个 observed、4 个 absent。两个人在 frame 0/cam4 各有 512 个可监督坐标点，visibility 可监督点数分别为 135 和 164。

## 训练时哪些信号会起作用

坐标与 uncertainty 一起计算损失。可见性只在标签为 `0` 或 `1` 且该点坐标有效时计算损失；`-1` 只跳过可见性损失，不自动取消可靠坐标的监督。人与人接触、地面接触没有训练标签和损失。原权重需要的接触头仍在模型里，但参数被冻结且不进入优化器；共享 decoder 若被更新，这些头的输出仍可能变化。

同一 episode、相同资产 ID 或相同源 manifest ID 不得同时出现在训练和验证集。这里不自动识别同一次动作录制的不同片段，也不增加额外分组任务。

当前训练配置是 `mamma_training_v2`：`loss_enabled.visibility` 是可见性损失及该头优化器成员的唯一开关。它现在为 `true`，但 `execution_authorized=false`，而可信遮挡负例资格仍是 `pending_representative_negative_evidence`。因此上面的 `plan`、`check-data` 可运行，**当前配置不能执行训练**；这次已有数据中的 visibility 计数为 604 个 `1`、0 个可信 `0`、3492 个 `-1`，不能据此宣称训练准备完成。

未来在数据资格和执行条件另行确认后，仍使用 `scripts/mamma_retrain.py train`，明确传入训练清单、互不交叉的验证清单、输出目录、设备、更新预算、checkpoint 间隔和 `--execute`。`evaluate` 也需要显式执行条件。正常恢复文件使用 `mamma_regular_resume_v3`，严格校验配置与数据合同，并保存模型、优化器、随机数状态、更新次数和下一个采样位置。当前没有用真实 MAMMA 网络验证新恢复路径。

想顺着代码阅读实现，请看 [源码阅读指南](CODE_GUIDE.md)。历史小试报告和产物保留作证据，不作为当前运行入口。
