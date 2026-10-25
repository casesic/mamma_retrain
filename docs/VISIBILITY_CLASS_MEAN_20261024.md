# 可见性类别均衡：一次固定的 500 步对照

这次只改了 visibility 损失怎样汇总点。原模型、普通 BCE 500 步模型和新模型使用同一批城市数据、同一取样顺序、AdamW 设置、原始权重和 0.5 判定阈值。新模型仍从原始权重开始，先训练 25 步，退出后从 checkpoint 在新进程恢复到累计 500 步；没有接着旧的 500 步模型训练。

## 为什么做这次实验

普通 BCE 微调后，模型把更多点判为可见。固定验证集上，可见点召回提高了，但遮挡点召回下降。先按训练真正使用的 `visibility_loss_valid` 统计，而不是拿全批标签数量估计：

| 训练范围 | observed 样本 | 可见监督点 | 遮挡监督点 | 两类都有的样本 | 仅可见 | 仅遮挡 | 无 visibility 监督 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 个完整训练场景 | 1628 | 245777 | 16581 | 126 | 1498 | 0 | 4 |
| 旧实验相同的前 500 次取样 | 500 | 74808 | 4454 | 33 | 466 | 0 | 1 |

前 500 条的训练类别分布：单人骑行 84 条，双人骑行 139 条，骑行＋步行 140 条，三人骑行 137 条。监督点数和旧训练日志逐条核对相同。损失先在每个人物样本内求均值，再对有监督的样本求均值；因此 74808:4454 不能直接当作总损失权重比。

普通模式 `point_mean` 对一条样本的所有已知点取 BCE 均值。新模式 `class_mean` 在一条样本同时有可见和遮挡点时取 `0.5 × mean(BCE_可见) + 0.5 × mean(BCE_遮挡)`；仅有一类时直接取该类均值。`-1` 不参与 visibility 损失。两类都没有时，该样本不参加 visibility 均值，但仍可用有效坐标监督。坐标＋uncertainty 损失、visibility 总系数、优化器、冻结头、阈值和数据均未改。由于前 500 条仅 33 条同时含两类，这种样本内平衡影响的范围本来就小。

## 固定验证集结果

验证集仍是四类各 v0005，共 435 条 observed、205173 个有效坐标点，visibility 真值为 66761 个可见、3771 个遮挡。下表“遮挡误判可见”越少越好；它不是对真实图片的准确率。

| 模型 | ERP 点均误差 px | ERP 样本等权误差 px | 可见召回 | 遮挡召回 | 遮挡误判可见 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 2.7597 | 3.1489 | 82.88% | 97.83% | 82 / 3771 |
| 普通 BCE 500 步 | 1.8336 | 2.1422 | 98.32% | 87.96% | 454 / 3771 |
| 类均衡 500 步 | 1.8333 | 2.1419 | 98.30% | 88.12% | 448 / 3771 |

相对普通 BCE，类均衡少误判 6 个遮挡点，但变化很小，没有恢复到原模型的遮挡召回。双人骑行的遮挡点坐标误差为原模型 8.4595、普通 BCE 8.5101、类均衡 8.5096 px，也基本没有变化。四类的可见/遮挡/unknown 坐标误差、分母和完整混淆计数见 [comparison.json](../outputs/experiments/city_full_visibility_class_mean_20261024/comparison.json)。单人骑行验证集没有遮挡真值，因此其遮挡召回没有分母。

四类单独看，下面每格按“原模型 / 普通 BCE / 类均衡”排列。点均和样本等权误差单位是完整 ERP 像素；遮挡点坐标误差另列，避免总体误差掩盖这部分目标。

| 验证类别（样本） | 点均误差 | 样本等权误差 | 可见召回 | 遮挡召回 | 遮挡点坐标误差 |
| --- | --- | --- | --- | --- | --- |
| 单人骑行（20） | 3.8723 / 2.1097 / 2.1092 | 3.8947 / 2.1750 / 2.1744 | 81.56% / 98.98% / 98.87% | 无遮挡 GT | 无遮挡 GT |
| 双人骑行（93） | 3.6585 / 2.7211 / 2.7206 | 4.9790 / 3.6640 / 3.6633 | 81.11% / 98.51% / 98.53% | 97.77% / 84.28% / 84.56% | 8.4595 / 8.5101 / 8.5096 |
| 骑行＋步行（129） | 3.4336 / 2.0613 / 2.0610 | 3.5435 / 2.2228 / 2.2226 | 84.21% / 98.20% / 98.18% | 98.08% / 88.57% / 88.77% | 9.0987 / 7.5884 / 7.5883 |
| 三人骑行（193） | 1.8168 / 1.2548 / 1.2546 | 1.9259 / 1.3517 / 1.3515 | 83.05% / 98.25% / 98.22% | 97.59% / 89.08% / 89.14% | 13.8795 / 10.9916 / 10.9929 |

## 产物与复现

实验配置是 [mamma_city_full_visibility_class_mean_20261024.json](../configs/retrain/mamma_city_full_visibility_class_mean_20261024.json)。[supervision_counts.json](../outputs/experiments/city_full_visibility_class_mean_20261024/supervision_counts.json) 记录训练侧统计；25、250、500 步 checkpoint 位于 `outputs/experiments/city_full_visibility_class_mean_20261024/train_25/` 和 `train_resume_500/`。实际 500 条 observation ID 与普通 BCE 实验逐条一致，未跳过更新；新进程恢复了第 25 步的模型、优化器、RNG 和取样位置。CPU 损失/梯度与恢复定向测试通过，真实 Blender、渲染和训练端之外的代码未运行。

实际在 GPU1（UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`）运行；训练循环用时约 4.57 秒（前 25 步）＋48.92 秒（续跑 475 步），固定验证集评价循环约 22.33 秒，另有每次进程的模型加载与数据检查用时。训练参与更新的四类数量分别是 84、139、140、137；16 个训练场景的逐场景次数列在 comparison.json。两个进程均以原始 `tensor_state.pt` 构造相同模型，续跑时再严格载入自己的第 25 步恢复状态。原权重的路径、文件大小和修改时间也记入 comparison.json 供核对，不涉及权重复制。

从 `mamma_retrain` 目录运行时，设置 `PYTHONPATH=.`、`PYTHON_DOTENV_DISABLED=1`、`PYTHONDONTWRITEBYTECODE=1`、`OPENBLAS_NUM_THREADS=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。模型命令还须显式选 GPU，并使用一个不存在的新输出目录：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
export PYTHONPATH=. PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
CONFIG=configs/retrain/mamma_city_full_visibility_class_mean_20261024.json
RUN_ROOT=outputs/experiments/city_full_visibility_class_mean_reproduction
CUDA_VISIBLE_DEVICES=1 "$MAMMA_PY" scripts/mamma_retrain.py train --config "$CONFIG" \
  --output "$RUN_ROOT/train_25" --device cuda:0 \
  --max-updates 25 --checkpoint-every 25 --execute
CUDA_VISIBLE_DEVICES=1 "$MAMMA_PY" scripts/mamma_retrain.py train --config "$CONFIG" \
  --output "$RUN_ROOT/train_resume_500" --device cuda:0 \
  --max-updates 500 --checkpoint-every 250 \
  --checkpoint "$RUN_ROOT/train_25/checkpoint_update_000025.pt" --execute
```

真实骑行图片仍无人工 512 点真值。新模型对冻结的 319 个目标补做推理，输入仿射、人物框、目标顺序与旧预测相同。固定 319×512 点上，原模型判可见 31,357 点，普通 BCE 判可见 79,594 点，类均衡判可见 80,677 点；普通 BCE 与类均衡在 1,209 个点上跨过 0.5 阈值，点位预测的中位周期距离差约 0.0048 px。[新预测 NPZ](../outputs/experiments/city_full_visibility_class_mean_20261024/predictions_real_class_mean_500.npz)和[页面用数据](../outputs/experiments/city_full_visibility_class_mean_20261024/review_predictions_class_mean_500.json)独立保存，旧两套预测未覆盖。人工页面可在双机整图切换三套模型，并以独立草稿评审“普通 BCE vs 类均衡”；旧“原模型 vs 普通 BCE”评分不会被迁移或覆盖。预测差异只说明模型输出不同，真实点位或遮挡是否变好，仍需用户看图判断。
