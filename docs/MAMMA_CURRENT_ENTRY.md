# MAMMA 当前使用入口

## 本轮已完成，先看这里

固定使用全量一遍的第812步权重与可见阈值 **0.85**，原模型以0.5保留对照。训练为28场景、3245条有效观察各一次、batch=4；主验证已统一为7场景814条。权重、配置校验值、运行命令及保存范围见 **[本轮交付说明](DELIVERY_20261025.md)**。

合成点位误差4.3954→2.5242 px；0.85下可见/遮挡召回为91.33%/94.76%。用户判断真实演示合图整体相近，共同成功35帧的可用点对2474→3665。当前没有替换部署权重，也不继续训练。

直接看图：在本项目目录用 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python scripts/real_stitch_review_server.py --host 127.0.0.1 --port 18872`，打开 `http://127.0.0.1:18872/`。合图链的范围和局限见[真实演示合图报告](REAL_RIDING_STITCH_THRESHOLD085_20261025.md)。

另有一项**原始 MAMMA 追加实验**：固定原模型阈值 0.5，在同一 40 对真实图片上比较直接合图与先做 SMPL-X 人体 fitting 再合图。它不使用上述微调权重，也不更改 0.85 微调交付。看图时运行 `/home/yy/conda_envs/camerahmr/bin/python scripts/fitting512/review_server.py --host 127.0.0.1 --port 18873`，打开 `http://127.0.0.1:18873/`；方法、结果和复现入口见[人体 fitting 对照说明](ORIGINAL_MAMMA_BODY_FITTING_20261025.md)。两组页面的人工评分分别保存。

## 检查入口及历史实验

下列16/4划分、500步和batch=1说明属于早期实验；普通CPU检查仍可使用禁止训练的mamma_regular.json。最终版本以以上交付入口为准。

首轮数据入口是相邻项目的 20 个完整城市场景，包内各有 `observations.json`、RGB、目标人物 mask 和有序 512 点标签。每个人有独立观察；`absent` 留在统计里，训练只取 `observed`。模型输入框从目标 mask 算出。点坐标保留完整 ERP 语义，跨左右接缝时裁剪和误差采用周期横坐标。

[固定划分](../configs/retrain/city_full_20261024_split.json)明确列出训练 16 个、验证 4 个数据包；每类的 v0005 作验证，其余四个完整场景作训练。`cycle_walk` 的训练场景为 v0001、v0002、v0004、v0006。路径逐包解析，不复制原图，也不把不同场景合成一个 episode。基础人物、动作和来源 manifest 可以在两侧复用；同一个 episode 或同一个数据包不能跨侧。

从项目目录运行，先只做 CPU 检查：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
export PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES='' "$MAMMA_PY" scripts/mamma_retrain.py plan \
  --config configs/retrain/mamma_regular.json --split train
CUDA_VISIBLE_DEVICES='' "$MAMMA_PY" scripts/mamma_retrain.py plan \
  --config configs/retrain/mamma_regular.json --split validation
CUDA_VISIBLE_DEVICES='' "$MAMMA_PY" scripts/mamma_retrain.py check-data \
  --config configs/retrain/mamma_regular.json --split validation --limit 2
```

`plan` 检查清单、包内文件、身份、时间和模板点序，不解码图片或加载模型。`check-data` 再读少量 RGB 和 mask。通用的 [mamma_regular.json](../configs/retrain/mamma_regular.json) 保持 `execution_authorized=false`；它不是可误触的训练配置。

本轮实际模型对照使用 [mamma_city_full_20261024.json](../configs/retrain/mamma_city_full_20261024.json)，只对这份经过代表性遮挡检查的完整批次授权执行。运行必须另加 `--execute`、显式输出新目录和设备。先在固定四场景验证集上评估原权重，再训练到第 25 次有效更新并保存 checkpoint；新进程从该 checkpoint 继续到累计 500 次，最终用**同一验证集**评估。命令和实际结果见 [本轮实验记录](CITY_FULL_FIRST_FINETUNE_20261024.md)。训练用 AdamW、学习率 1e-5、batch size 1；计划中的 500 指真正执行优化器的次数，空监督样本不计入。

另有一份[可见性类均衡 500 步配置](../configs/retrain/mamma_city_full_visibility_class_mean_20261024.json)，它只把 visibility 的逐样本 BCE 归约从 `point_mean` 改为 `class_mean`。固定数据、三模型指标和 checkpoint 见[类均衡对照记录](VISIBILITY_CLASS_MEAN_20261024.md)。历史 checkpoint 只作经过来源核对的模型权重读取，不接入新实验的优化器或恢复状态。

当前的[30% 遮挡取样配置](../configs/retrain/mamma_city_full_occlusion_30pct_20261025.json)沿用类均衡损失，从原始权重另训 500 次；固定顺序在[取样计划](../outputs/experiments/city_full_occlusion_30pct_20261025/sampling_plan.json)。每十次更新安排三条有有效遮挡监督的训练观察，其余七条来自其他观察；不会改变验证集分布。四模型指标、命令和局限见[30% 取样对照记录](OCCLUSION_30PCT_20261025.md)。普通检查依旧使用禁止执行的 `mamma_regular.json`，不会自动训练。

新一轮[合并数据配置](../configs/retrain/mamma_city_full_plus_occlusion_30pct_20261025.json)把原训练 16 场景与新遮挡训练 12 场景合并，仍从原始权重进行 500 次 `class_mean`、30% 遮挡取样更新。主要看它相对**未微调原模型**的效果：两组验证的点位误差明显下降，可见召回上升，遮挡召回下降；[报告](CITY_MERGED_OCCLUSION_30PCT_20261025.md)先给这组主对照，再用旧数据 30% 模型分析新增数据的额外作用。旧 435 条验证与新 379 条遮挡验证分开评价。跨验证集只读评价可重复传入 `--eval-manifest` 指定新包；训练恢复仍严格绑定原配置、数据和取样计划。

现行[完整一遍配置](../configs/retrain/mamma_city_merged_one_epoch_20261025.json)仍从原权重开始，但把合并训练集里有监督的 3245 条观察各训练一次，不设遮挡配额。它使用真实 batch 4，共 812 次优化器更新，尾批 1 条。现在以原 4 个加新增 3 个场景组成的**统一 814 条验证集**为主结果；[统一划分](../configs/retrain/city_merged_unified_validation_20261025_split.json)供以后新实验直接使用。[训练与对照报告](CITY_MERGED_FULL_EPOCH_20261025.md)显示相对原模型，ERP 点均误差 4.3954→2.5242 px、可见召回 82.64%→98.42%，但遮挡召回 95.73%→85.94%。统一[对照数据](../outputs/experiments/city_merged_one_epoch_20261025/combined_validation_comparison.json)直接从逐样本预测重算，保留七份包和各类明细；完成一遍后已停止，没有自动进入第二轮。

坐标与 uncertainty 联合学习。visibility 只用 `0/1`，跳过 `-1`；未知或遮挡不会自动取消可靠坐标监督。人与人接触和地面接触不作为标签或损失，原权重所需的对应头保留并冻结。基于 [遮挡抽查](../outputs/experiments/city_full_first_finetune_20261024/OCCLUSION_CHECK.md) 开放的 visibility 资格只适用于本次有界实验，不代表全批逐点人工认证或真实拍摄泛化。

源码入口与修改位置见 [CODE_GUIDE.md](CODE_GUIDE.md)。

真实骑行图片没有人工 512 点真值。同帧 cam2/cam4 的[点位人工对照](REAL_RIDING_MANUAL_REVIEW_20261024.md)现以“原模型阈值 0.5 vs 合并数据全量一遍阈值 0.85”为默认独立评审组，旧 0.5 评分保留。想直接看最终画面，可按[真实双机演示合图说明](REAL_RIDING_STITCH_THRESHOLD085_20261025.md)启动本地页面，并排查看固定 40 对的原模型与 0.85 方案。用户目前的总体画面判断是两者效果相近，0.85 方案可用点更多；这不等于真实点位准确率的真值评价。当前运行中的旧服务不会自动加载新页面代码。

另有一次独立的[自遮挡 unknown 补负例试验](SELF_OCCLUSION_UNKNOWN_TRIAL_20261025.md)：仅在身体射线与保存的渲染前景双重证据满足保守规则时，将试验包中的 -1 改为 0。它复用同一 28/7 场景、同一可训练观察顺序，从原权重重新训练一遍；旧包、默认标签规则、现有交付模型均未替换。报告给出同一旧标签验证集上的三模型指标及固定 319 个真实目标的独立看图入口。真实图没有人工真值，新的点位对照仍需用户判断。
