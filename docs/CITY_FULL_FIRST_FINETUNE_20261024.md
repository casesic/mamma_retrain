# 首轮城市完整场景微调：500 次更新的同集对照

## 初步结果摘要（2026-10-24）

**初步结论：在同资源池的新合成场景上，500 步微调降低了整体点位误差，提高了可见点召回，但遮挡点召回下降。真实拍摄画面上的收益尚待人工对照判断。**

- 数据：20 个完整城市场景，固定 16 个训练、4 个验证；训练 1,628 条 observed，验证 435 条 observed。旧三帧小样全部排除。同一场景不跨集，基础人物、动作、衣物、发型及来源允许复用。
- 训练：RGB＋目标 mask 输入；有序 512 点坐标、uncertainty 与 visibility 联合学习，接触任务关闭。AdamW，学习率 1e-5，batch size 1，共 500 次有效更新；第 25 步退出后在新进程恢复并继续完成。
- 对照：原模型与第 500 步模型使用相同的 435 条验证观察、205,173 个有效坐标点。以下为合成验证指标，不是实际拍摄图片上的精度。

| 指标 | 原模型 | 500 步后 | 变化 |
| --- | ---: | ---: | ---: |
| 有效点平均 ERP 误差 | 2.76 px | 1.83 px | 相对下降约 33.6% |
| 可见点召回 | 82.88% | 98.32% | 上升 15.44 个百分点 |
| 遮挡点召回 | 97.83% | 87.96% | 下降 9.86 个百分点 |

变化量根据未四舍五入的原始数值计算。四类场景的总体坐标误差均下降；双人骑行类的遮挡点坐标误差略升（8.46→8.51 px）。因此不能概括为所有指标均改善。可见性指标仅统计坐标有效且标签为 0/1 的点，分母分别为 66,761 个可见点、3,771 个遮挡点；unknown 不参与可见性评价。

这是单次 500 步实验，未检验不同随机种子的稳定性，也未证明对未见资产或真实画面的泛化。验证集没有实际跨 ERP 接缝目标。遮挡标签完成代表性目视和保存证据一致性检查，不等于逐点人工认证。

下一步已交接真实图片人工对照：固定真实图片与人物 mask，比较原模型和微调模型的点位与遮挡判断。真实图片没有配套人工真值，效果需由人工评审，当前不记录为已提升。详细指标、复现命令及 checkpoint 入口见下文；本摘要按已保存的 comparison.json 核对，未重跑模型。

## 实验数据与模型

本轮只用 [完整城市批次索引](../../human_data_generator/outputs/batches/city_full_20261024/index.json) 指向的成功包，没有混入旧三帧样本或失败目录。固定划分写在 [city_full_20261024_split.json](../configs/retrain/city_full_20261024_split.json)：每类 v0005 验证，其他四个训练；骑行＋步行训练使用 v0001、v0002、v0004、v0006。训练为 16 个 episode、1628 条 observed 和 932 条 absent；验证为 4 个 episode、435 条 observed 和 205 条 absent。episode 没有交叉。两侧有 4 个相同的来源 manifest ID，符合本轮允许复用基础来源的决定。

[遮挡抽查](../outputs/experiments/city_full_first_finetune_20261024/OCCLUSION_CHECK.md)目视检查了三类多人场景各两个含 `0` 的观察，并检查全批 24,194 个负例的保存证据一致性。它支持**本次实验**使用当前 visibility 监督，不表示所有点都经过人工确认。原权重来自 [tensor_state.pt](../outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt)（540,487,469 字节，2026-10-15 17:44 本地修改时间）；原模型和最终评价都严格加载 282/282 个同名同形状 tensor，无缺失或额外键。没有换成随机初始化。

## 实际结果

下表的误差是完整 ERP 上所有有效点的**按点平均欧氏误差**，横向跨缝按最短周期距离计算。召回只用坐标有效且 visibility 为 `0/1` 的点；`-1` 不参加可见性分母。前后评价逐条身份和顺序一致，共 435 条 observed。

| 场景类型 | 样本 / 有效点 | ERP 误差 原→500 步 | 可见召回 原→500 步 | 遮挡召回 原→500 步 |
| --- | ---: | ---: | ---: | ---: |
| 单人骑行 | 20 / 9,167 | 3.87→2.11 px | 81.56%→98.98% | 无负例 |
| 双人骑行 | 93 / 43,496 | 3.66→2.72 px | 81.11%→98.51% | 97.77%→84.28% |
| 骑行＋步行 | 129 / 58,453 | 3.43→2.06 px | 84.21%→98.20% | 98.08%→88.57% |
| 三人骑行 | 193 / 94,057 | 1.82→1.25 px | 83.05%→98.25% | 97.59%→89.08% |
| **全部** | **435 / 205,173** | **2.76→1.83 px** | **82.88%→98.32%** | **97.83%→87.96%** |

验证集可见性监督分母为 66,761 个 `1` 和 3,771 个 `0`。按点的坐标误差：可见点 1.77→1.37 px，遮挡点 10.93→9.15 px，unknown 点 3.02→1.86 px。unknown 的坐标仍可监督；它没有被当成 visibility 负例。样本等权平均误差为 3.15→2.14 px。双人骑行类的遮挡点坐标误差单独看略升（8.46→8.51 px）；三类含负例场景的遮挡召回均下降。不能把总体误差下降写成所有指标都改善。

[一页看图](../outputs/experiments/city_full_first_finetune_20261024/inspection.html)使用在最终评价前固定的四条样本，逐行保持同一原图、裁剪范围与点序，展示 GT、原模型和 500 步结果；[comparison.json](../outputs/experiments/city_full_first_finetune_20261024/comparison.json)保留精确分母与四类数值。验证四场景没有实际跨 ERP 左右边界的目标，因此无法从这份验证集评价跨缝预测；另有训练包的跨缝目标完成只读 reader 还原检查，最大误差约 0.00012 px。

## 训练与恢复

训练使用单张 RTX 3090：GPU0，UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，与其他进程共享，未停止别人的任务。AdamW 学习率 1e-5、batch size 1、固定 sampler seed 20261024。混合顺序每轮从各 episode 轮流取样；前 16 条覆盖全部训练场景。500 条记录全部为有效优化器更新，无空监督跳过，16 个场景各参与 20–35 次。第 1 次的 query 参数梯度最大绝对值 0.738，参数更新最大绝对值约 1.00e-5；各检查点无非有限 loss/梯度。接触与地面接触头均未改变。

先运行 25 次并退出进程，保存 [第 25 步 checkpoint](../outputs/experiments/city_full_first_finetune_20261024/train_25/checkpoint_update_000025.pt)。新进程严格恢复模型、AdamW、Python/NumPy/PyTorch RNG、epoch 和采样位置；续训首条正是第 26 步、位置 25。随后保存 [第 250 步](../outputs/experiments/city_full_first_finetune_20261024/train_resume_500/checkpoint_update_000250.pt)和[第 500 步 checkpoint](../outputs/experiments/city_full_first_finetune_20261024/train_resume_500/checkpoint_update_000500.pt)。两段训练本体分别耗时约 5.9 秒和 70.7 秒；最终 435 条评价耗时约 21.1 秒，不含模型构造、数据合同检查和 checkpoint 读写。前 25 步平均 loss 为 -777.52，末 25 步为 -902.48；负值来自当前坐标 log-sigma 损失公式，不能单独解释为实际像素精度。

本轮配置是 [mamma_city_full_20261024.json](../configs/retrain/mamma_city_full_20261024.json)，通用配置仍禁止执行。如下命令从项目目录、现有 MAMMA Python 环境运行；真实执行必须显式加 `--execute`、新输出目录和设备。运行前设置 `PYTHON_DOTENV_DISABLED=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OPENBLAS_NUM_THREADS=1`，并将 `CUDA_VISIBLE_DEVICES=0` 映射到上述 UUID：

```bash
PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
CFG=configs/retrain/mamma_city_full_20261024.json
BASE=outputs/experiments/city_full_first_finetune_20261024
$PY scripts/mamma_retrain.py evaluate --config "$CFG" --split validation --output "$BASE/baseline_eval_retry1" --device cuda:0 --execute
$PY scripts/mamma_retrain.py train --config "$CFG" --output "$BASE/train_25" --device cuda:0 --max-updates 25 --checkpoint-every 25 --execute
$PY scripts/mamma_retrain.py train --config "$CFG" --output "$BASE/train_resume_500" --device cuda:0 --max-updates 500 --checkpoint-every 250 --checkpoint "$BASE/train_25/checkpoint_update_000025.pt" --execute
$PY scripts/mamma_retrain.py evaluate --config "$CFG" --split validation --output "$BASE/final_eval" --device cuda:0 --checkpoint "$BASE/train_resume_500/checkpoint_update_000500.pt" --execute
```

上面展示的是本次实际参数；**重跑时必须换新的输出目录名**，程序不会覆盖已有结果。原模型第一次尝试暴露 CLI 未设置本仓库上游 `landmarks` 导入路径，第二次在受限执行环境看不到 CUDA，第三次缺少确定性 cuBLAS 工作区设置；对应失败日志和未完成目录留在同一实验目录。修复仅在自写模型加载入口与数值环境设置，未改上游 MAMMA 文件。最终基线、25+475 次训练和最终评价均成功。本轮未做真实拍摄画面测试、跨缝验证集对照或进一步超参数搜索，也未将此 checkpoint 当作可直接部署模型。
