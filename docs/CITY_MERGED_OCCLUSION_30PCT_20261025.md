# 合并遮挡场景后，与原模型的 500 步对照

当前主验证口径已将原有 4 场景与新增遮挡 3 场景合为一份 814 条观察的验证集；原模型、这份 500 步模型和全量一遍模型的统一结果见[合并验证对照](../outputs/experiments/city_merged_one_epoch_20261025/combined_validation_comparison.json)。下方保留的是当时分别核查两类场景的历史记录。

主要问题是：合并遮挡数据微调后，比**未微调的原模型**好吗？两组验证集都显示坐标点位明显更准、可见点漏判更少；代价是遮挡点更容易被误判为可见。与上一轮“旧数据＋30%取样”模型的比较只用于判断新增数据是否带来额外收益，不替代原模型基线。完整数值在[对照结果](../outputs/experiments/city_full_plus_occlusion_30pct_20261025/comparison.json)。

本次训练使用原来的 16 个城市场景加上 12 个新遮挡场景；从同一原始权重开始，使用 `class_mean` 可见性损失、每十步三条遮挡观察、AdamW 学习率 `1e-5`，训练 500 次有效更新。旧的 435 条验证观察和新遮挡的 379 条验证观察分别评价，均未参与取样。

[合并划分](../configs/retrain/city_full_plus_occlusion_20261025_split.json)列出 28 个训练包和旧验证 4 包；[新遮挡划分](../../human_data_generator/outputs/batches/occlusion_rules_five_20261025/new_occlusion_split.json)单独列出新验证 3 包。35 个 episode 互不重复、互不跨侧。训练池 3255 条 observed 中，953 条有实际 `visibility_loss_valid` 内的遮挡点，2302 条没有；后者有 10 条完全没有可更新监督（旧数据 1 条、新数据 9 条），只保留在统计中。[冻结计划](../outputs/experiments/city_full_plus_occlusion_30pct_20261025/sampling_plan.json)取 150 条遮挡、350 条其他观察，500 条均不重复，覆盖全部 28 个训练 episode。实际更新来自旧场景 273 次、新场景 227 次；这是现有取样算法的结果，没有额外按来源配额。

前 25 步退出进程，之后从[第 25 步 checkpoint](../outputs/experiments/city_full_plus_occlusion_30pct_20261025/train_25/checkpoint_update_000025.pt)在新进程续至[第 500 步](../outputs/experiments/city_full_plus_occlusion_30pct_20261025/train_resume_500/checkpoint_update_000500.pt)，另保存第 250 步。第 26 条与冻结计划一致；500 条日志逐条匹配，150/350 次、零跳过，接触头未更新，原权重严格加载。训练在 GPU0 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 执行，两个训练循环分别用时约 4.37 和 51.11 秒；不含进程启动与模型加载。独立评价、真实推理使用 GPU0/GPU1，未操作其他进程。

下表只在**同一组验证集内**比较。点均误差按有效点计，样本等权误差先对每条观察求均值；遮挡召回的分母分别是旧验证 3771 点、新验证 17985 点。原模型与合并数据模型是主要两行，旧数据 30% 模型是附加对照。

| 验证集 | 模型 | ERP 点均误差 px | 样本等权 px | 可见召回 | 遮挡召回 | 遮挡误判可见 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 旧 435 条 | 原模型 | 2.7597 | 3.1489 | 82.88% | 97.83% | 82 / 3771 |
| 旧 435 条 | 合并数据 30% | 1.8414 | 2.0348 | 97.84% | 87.51% | 471 / 3771 |
| 旧 435 条 | 旧数据 30%（附加） | 1.7835 | 2.1972 | 97.74% | 88.52% | 433 / 3771 |
| 新 379 条 | 原模型 | 6.3346 | 7.7877 | 82.30% | 95.29% | 847 / 17985 |
| 新 379 条 | 合并数据 30% | 4.4017 | 5.6576 | 97.32% | 87.33% | 2279 / 17985 |
| 新 379 条 | 旧数据 30%（附加） | 4.2801 | 5.6309 | 97.07% | 88.33% | 2098 / 17985 |

相对原模型，旧验证的点均误差从 2.76 降至 1.84 px、新验证从 6.33 降至 4.40 px；可见召回两组都从约 82% 升至约 97%。遮挡召回则分别从 97.83% 降至 87.51%、从 95.29% 降至 87.33%，误判可见的遮挡点由 82 增至 471、由 847 增至 2279。可见性有明显取舍，不能只用点位误差说整体全面提升。

次要问题是新增数据的额外收益：合并数据模型相对旧数据 30% 模型在两组验证的总体点均误差和遮挡召回都退化；旧验证的样本等权误差改善了约 0.16 px。新验证的双人直行类别点均误差和遮挡召回略有改善，三人直行退化较明显，双人斜交点位略好但遮挡召回略差。完整的四旧类、三新类以及按 GT 可见／遮挡／未知分组的误差和混淆分母在对照 JSON 中。500 步短程结果不等于最终训练结论。

对固定真实骑行集的 319 个目标只补推[本轮模型预测](../outputs/experiments/city_full_plus_occlusion_30pct_20261025/predictions_real_class_mean_merged_occlusion30_500.npz)。目标顺序、模板、仿射和输入指纹与旧数据 30% 模型逐数组一致；原模型预测可见点为 31,357，本轮模型为 89,979。真实图片没有人工 512 点真值，可见数量增加不等于可见性更正确。在[实名双机人工对照说明](REAL_RIDING_MANUAL_REVIEW_20261024.md)的页面中，默认打开“原模型 vs 新增遮挡数据 30%”独立评审；“旧数据 30% vs 新增遮挡数据 30%”保留为次要对照。两组都初始未评审，不迁移旧评分。**现有服务不会自动加载新页面代码**；用户稍后可在项目目录运行 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python scripts/real_review_server.py --host 127.0.0.1 --port 18869` 并打开 `http://127.0.0.1:18869/`，或在结束自己的旧服务后重新启动原端口。本轮没有重启用户服务。

复现时从 `mamma_retrain` 目录使用 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`，设置 `PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`。训练配置为[本轮配置](../configs/retrain/mamma_city_full_plus_occlusion_30pct_20261025.json)，先以 `scripts/freeze_occlusion30_plan.py --config <本轮配置>` 冻结计划（已有计划不可覆盖），再用 `scripts/mamma_retrain.py train --config <本轮配置> --output <新目录> --device cuda:0 --max-updates 25 --checkpoint-every 25 --execute`；第二次加 `--checkpoint <第25步文件> --max-updates 500 --checkpoint-every 250`。旧验证用 `evaluate --split validation`；新验证另加三次 `--eval-manifest <新遮挡划分中的实际包路径>`。只读评价会核对 checkpoint 的原训练合同，同时把本次评价清单单独记入结果，不篡改恢复状态。

例如，以下命令只读评价本轮最终模型在新遮挡验证集上的表现；`--output` 必须换成尚不存在的新目录：

```bash
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
export PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG=configs/retrain/mamma_city_full_plus_occlusion_30pct_20261025.json
RUN=outputs/experiments/city_full_plus_occlusion_30pct_20261025
CUDA_VISIBLE_DEVICES=0 "$MAMMA_PY" scripts/mamma_retrain.py evaluate \
  --config "$CONFIG" --split validation --device cuda:0 --execute \
  --output outputs/experiments/merged30_new_validation_recheck \
  --checkpoint "$RUN/train_resume_500/checkpoint_update_000500.pt" \
  --eval-manifest ../human_data_generator/outputs/runs/occlusion_five_20261025_straight2_v0005/observations.json \
  --eval-manifest ../human_data_generator/outputs/runs/occlusion_five_20261025_diagonal2_v0005/observations.json \
  --eval-manifest ../human_data_generator/outputs/runs/occlusion_five_20261025_straight3_v0005/observations.json
```

CPU 现行 13 项测试通过；合并训练／两组验证的身份、模板和 GT 对齐、500 条取样日志以及真实集指纹均逐项核对；把训练 episode 传作外部验证会在模型加载前被拒绝，错误 checkpoint 合同同样拒绝。临时 Chrome 打开新页面并加载新增预测数据与双机原图；隔离浏览器资料中实测“原模型 vs 本轮模型”默认打开、两个新增组初始空白、评分保存、导出再导入、错组导入拒绝及旧组评分隔离，测试资料不在交付页面中。本轮未填写用户的实际人工判断。没有渲染、扩批、改标签、训练第二个比例、更新部署权重或提交 Git。
