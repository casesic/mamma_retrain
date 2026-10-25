# 合并数据完整训练一遍：与原模型比较

这次回答的问题是：把现有训练观察各用一次后，模型相对**未微调的原模型**怎样变化？此前的合并数据 500 步模型仅作附加参考。现在把原有 4 个城市验证场景与新增 3 个遮挡验证场景作为**同一份 7 场景、814 条观察的验证集**，统一计算主结果。

训练数据是[合并划分](../configs/retrain/city_full_plus_occlusion_20261025_split.json)中的 28 个场景、3255 条 observed。按当前 reader 的真实监督掩码，其中 3245 条有坐标或可见性监督，另外 10 条两项都没有。冻结的[一次完整顺序](../outputs/experiments/city_merged_one_epoch_20261025/sampling_plan.json)记录了每条观察的身份、有效监督点数和这 10 条排除记录；可训练观察各出现一次，含有效遮挡监督的为 953 条。验证数据未进入顺序。

正式设置由[训练配置](../configs/retrain/mamma_city_merged_one_epoch_20261025.json)固定：从原始 tensor_state 严格加载，`class_mean` 可见性损失，AdamW 学习率 `1e-5`，真实 batch size 4，FP32，不使用梯度累积、额外遮挡配额或重复取样。每个 batch 对坐标和可见性分别按**有该任务监督的样本**求均值；最后不足四条的 batch 照常更新。接触头保留并冻结，visibility 的 `-1` 不进入可见性损失。

正式训练前，独立从原权重执行三个 disposable batch，含遮挡样本和仅有坐标监督的样本；[显存短测](../outputs/experiments/city_merged_one_epoch_20261025/memory_probe_batch4.json)显示 peak allocated 3.40 GB、peak reserved 3.75 GB，三次更新耗时 1.44 秒（不含模型加载）。短测进程退出后，正式训练重新初始化模型、优化器与随机种子。正式输出位于[训练目录](../outputs/experiments/city_merged_one_epoch_20261025/train_full/)；每次更新记录该 batch 的全部 observation ID、batch 大小、样本游标和分项损失。checkpoint 只在完整优化器更新后保存，恢复绑定训练来源、冻结顺序、batch size、优化器、RNG、更新数和下一样本位置。

正式训练在 GPU0 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 完成 812 次更新，训练循环耗时 247.6 秒（不含模型初始化和数据预检）。其中 811 个四条 batch、最后一个一条；展开更新日志后与冻结计划的 3245 个 observation ID **逐条一致且无重复**，28 个训练 episode 均覆盖。分项损失均有限，接触头在训练前后逐张量一致。[第 500 步](../outputs/experiments/city_merged_one_epoch_20261025/train_full/checkpoint_update_000500.pt)只是中间恢复点；预先指定的评价模型是[第 812 步最终 checkpoint](../outputs/experiments/city_merged_one_epoch_20261025/train_full/checkpoint_update_000812.pt)。本次没有中断正式训练；严格恢复的样本位置和 batch 合同通过小型测试，不能据此声称实际执行过断点续训。

## 验证结果

**原模型→完整一遍**是主对照；合并数据 500 步是附加参考。[统一划分](../configs/retrain/city_merged_unified_validation_20261025_split.json)明确列出 28 个训练场景与 7 个验证场景，供后续实验直接使用。点均误差以全部 378,233 个有效坐标点为分母，样本等权误差先按人物观察求均值。可见／遮挡召回分别以 111,917 个可见点和 21,756 个遮挡点为分母，阈值均为 0.5。这里直接拼接保存的逐样本预测、GT 与监督掩码后重算，**没有平均两个旧汇总分数**。7 份 manifest、同一 814 条观察的顺序、模板点序和三模型的 GT／掩码均经核对；完整分类指标见[统一对照数据](../outputs/experiments/city_merged_one_epoch_20261025/combined_validation_comparison.json)。

| 模型 | 点均误差 px | 样本等权 px | 可见召回 | 遮挡召回 | 遮挡误判可见 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 4.3954 | 5.3026 | 82.64% | 95.73% | 929 / 21756 |
| 完整一遍 | 2.5242 | 3.0956 | 98.42% | 85.94% | 3059 / 21756 |
| 合并数据 500 步（参考） | 3.0129 | 3.7168 | 97.63% | 87.36% | 2750 / 21756 |

完整一遍让统一验证集的点位误差和可见点漏判明显下降，但遮挡误判可见从 929 增至 3059。这是明确的代价，不能只据点位误差说模型全面更好。相对 500 步模型，点位继续改善，遮挡召回继续下降；两者训练样本数量、顺序和真实 batch 都不同，不是严格单变量实验。七种场景类别及按 GT 可见／遮挡／unknown 分组的坐标误差也在统一对照数据中；其中单人骑行没有遮挡标签，遮挡召回无分母。旧的[分组验收结果](../outputs/experiments/city_merged_one_epoch_20261025/comparison.json)只作历史核查，不再作为主表。

用户随后选择将**这份微调权重的可见判定阈值设为 0.85**，原模型仍用 0.5；权重、logits、概率和坐标都未改。用上述 814 条保存的逐点预测与同一有效掩码重算，0.85 下 111,917 个真实可见点判对 102,209 个，可见召回 **91.33%**；21,756 个真实遮挡点判对 20,617 个，遮挡召回 **94.76%**，遮挡误判可见 1,139 个。点均误差保持 2.5242 px。这一阈值是在查看本验证集后确定的，属于**同集选择后的描述性结果**，不是独立测试证实；真实图片仍没有可见性真值。既有 0.5 指标和人工评审身份保留，不把旧评分迁到 0.85。

## 复现入口

在 `mamma_retrain` 目录使用 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`，设置 `PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`，并设置 `PYTHONPATH=.:src`。新实验使用独立输出目录，已有目录不能覆盖。

先运行 `python -m scripts.freeze_one_epoch_plan --config configs/retrain/mamma_city_merged_one_epoch_20261025.json --output outputs/experiments/city_merged_one_epoch_20261025/sampling_plan.json` 冻结一次顺序；已有计划不可重写。用 `scripts/probe_one_epoch_batch.py` 的 `--batch-size 4`、`--device cuda:0` 和独立 `--output` 做短测。正式训练命令为：

```bash
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
CUDA_VISIBLE_DEVICES=<实际 GPU UUID> "$MAMMA_PY" scripts/mamma_retrain.py train \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --device cuda:0 --max-updates 812 --checkpoint-every 500 \
  --output outputs/experiments/city_merged_one_epoch_20261025/train_full --execute
```

`--max-updates` 是 optimizer 更新数，样本数由冻结顺序决定。完整训练时按实际 3245 条和 batch 4 推得 812 次更新；最后一批只有一条。若从中间 checkpoint 续训，应使用**新的**输出目录并带 `--checkpoint`，不能把已经消费的样本再作为训练起点。

真实骑行 319 个目标没有人工 512 点真值。新模型只在同一批原图、mask、裁剪和仿射上补推理；模型间点位差和预测可见点数量只供人工查看，不能当真实误差或召回。

本轮[真实预测](../outputs/experiments/city_merged_one_epoch_20261025/predictions_real_class_mean_merged_full_epoch.npz)覆盖固定的 319/319 目标，输入指纹、目标顺序、模板点序和仿射与原有预测一致。新模型预测可见 105,560 点，原模型为 31,357 点；这个差异**不是**真实准确率。在[实名双机人工对照页面](REAL_RIDING_MANUAL_REVIEW_20261024.md)中，默认新增“原模型 vs 合并数据全量 1 遍”，也可单独选“合并数据 500 步 vs 全量 1 遍”。两组评分身份独立，初始均未评审；旧评分未迁移或改写。用户稍后可以在项目目录运行：

```bash
/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python scripts/real_review_server.py \
  --host 127.0.0.1 --port 18871
```

再打开 `http://127.0.0.1:18871/`。端口若被自己的其他服务占用，请选一个空闲端口；已有服务不会自动载入更新后的页面，本轮没有重启它。

原 4 份与新增 3 份验证的模型预测已经保存；现在运行 `python -m scripts.full_epoch_compare --combined`，从这些逐样本预测统一计算 814 条的结果，不重复加载模型或图片。该命令核对[统一划分](../configs/retrain/city_merged_unified_validation_20261025_split.json)、目标顺序、模板、GT 与掩码，并只创建一次[统一对照数据](../outputs/experiments/city_merged_one_epoch_20261025/combined_validation_comparison.json)。已有 checkpoint 的严格恢复合同仍绑定训练当时的旧验证引用；因此本次另存统一划分，合并统计不会改写 checkpoint、原始包或历史分组结果。未来新实验可直接将统一划分用作其 `input_config`。

本轮定向 16 项 CPU 测试通过；真实 batch=4 显存短测、完整训练、两组真实模型评价和 319 目标推理已执行。临时回环服务中的独立 Chrome 资料实测页面加载、默认未评审、组间评分隔离、导入导出往返及错误组导入拒绝；用户原服务和浏览器资料未操作。没有进行第二轮训练、真实图标注、部署权重替换或渲染。合成验证的改善不能直接证明真实拍摄画面点位更准，真实表现仍要由用户在同一输入上人工判断。
