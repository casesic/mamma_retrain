# 30% 遮挡取样：固定 500 步对照

这次从**原始 MAMMA 权重**重新训练，继续使用上一轮的 `class_mean` 可见性损失，只改变训练样本出现的顺序与频率。500 次有效更新中，150 次选有有效遮挡监督的观察，350 次选其他观察。验证仍用原来的四个完整场景，没有重采样。新场景、真实图片标签和编辑器规则均未进入训练。

## 取样到底改变了什么

训练集有 1628 条 `observed`：126 条在实际 `visibility_loss_valid` 内含 `0`，1502 条不含。四类训练场景的遮挡池数量分别是双人骑行 55、骑行＋步行 35、三人骑行 36；单人骑行没有遮挡监督。1502 条其余观察中有 1 条既无有效坐标也无可见性监督，因此仍留在池的统计里，但不会被选作一次有效更新。

[固定取样计划](../outputs/experiments/city_full_occlusion_30pct_20261025/sampling_plan.json)用 seed `20261024` 一次生成，并以 `3003c2a8febbe740cc9fa4693ed483c9` 标识。每十次更新恰好三条来自遮挡池，池内按 episode 分散、确定性打乱，先耗尽才重复。实际用了遮挡池 150 次（126 条各至少一次、其中 24 次必要重复）和其余池 350 次（350 条不同观察）；16 个训练 episode 全部参与。遮挡池的这 150 次携带 19,880 个有效 `0` 点。由于同一样本的坐标损失也更常出现，这一影响属于取样策略本身，没有另加坐标权重。

原权重严格匹配 282/282 个模型键；AdamW、学习率 `1e-5`、batch size 1、模型种子、标签、裁剪、`class_mean` 公式和 0.5 可见阈值与上一轮相同。GPU0 UUID 为 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`。前 25 步退出进程，随后从[第 25 步 checkpoint](../outputs/experiments/city_full_occlusion_30pct_20261025/train_25/checkpoint_update_000025.pt)在新进程恢复至累计 500 步；第 26 条与预存计划一致。还保存了[第 250 步](../outputs/experiments/city_full_occlusion_30pct_20261025/train_resume_500/checkpoint_update_000250.pt)和[第 500 步](../outputs/experiments/city_full_occlusion_30pct_20261025/train_resume_500/checkpoint_update_000500.pt)。500 条训练日志与计划逐条一致，零跳过。训练循环用时约 4.41＋49.78 秒，验证循环约 20.86 秒；这些数值不含各进程加载模型和数据的时间。

## 同一验证集的结果

四套模型严格对齐同一 435 条验证观察、512 点模板、205173 个有效坐标点和 66761／3771 个可见／遮挡监督点。下表“遮挡误判可见”越少越好；样本等权误差给每条观察相同权重，点均误差给每个有效点相同权重。

| 模型 | ERP 点均误差 px | ERP 样本等权 px | 可见召回 | 遮挡召回 | 遮挡误判可见 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 2.7597 | 3.1489 | 82.88% | 97.83% | 82 / 3771 |
| 普通 BCE 500 步 | 1.8336 | 2.1422 | 98.32% | 87.96% | 454 / 3771 |
| 类均衡 500 步 | 1.8333 | 2.1419 | 98.30% | 88.12% | 448 / 3771 |
| 类均衡＋30% 遮挡取样 | **1.7835** | 2.1972 | 97.74% | **88.52%** | **433 / 3771** |

相对直接对照的类均衡模型，遮挡误判可见减少 **15 个点**，但可见召回下降约 **0.56 个百分点**。点均误差下降，样本等权误差上升，不能只报一个数说点位全面改善。双人骑行遮挡召回 84.56%→86.09%，遮挡点坐标误差 8.51→7.13 px；三人骑行遮挡召回 89.14%→88.82%，反而略降。按类别及可见／遮挡／unknown 分组的分母、误差和混淆数在[完整对照](../outputs/experiments/city_full_occlusion_30pct_20261025/comparison.json)。单人骑行验证场景没有遮挡 GT，因此没有该类遮挡召回。

这是一次预先确定的 30% 比例试验，不是比例搜索，也没有产生新的遮挡多样性。遮挡池仅 126 条，必须重复 24 次；效果变化不能解释为新场景带来的提升。`-1` 仍不参与 visibility 损失，坐标有效时仍训练坐标。

## 真实图片怎么看

新模型只对原冻结的 40 对 ERP、80 张原图、319 个 target 补做推理，没有重新检测、换 mask 或换人物。与上轮预测的目标顺序、模板点序、框仿射及输入裁剪指纹全部相同。固定 319×512 个预测点上，类均衡模型判可见 80,677 个，本模型判可见 90,070 个；其中 10,802 个从“不判可见”变为“判可见”，1,409 个反向变化。这些是**模型间预测差异，不是实际正确率或召回**。真实图片无人工 512 点真值，尤其遮挡处不能仅凭可见点数量判断好坏。

[原实名双机人工页面](REAL_RIDING_MANUAL_REVIEW_20261024.md)现在可切换第四套模型，并新增“类均衡 vs 类均衡＋30% 遮挡取样”独立评审组。新组初始未评审，旧两组评分不迁移。新[预测数组](../outputs/experiments/city_full_occlusion_30pct_20261025/predictions_real_class_mean_occlusion30_500.npz)和[页面数据](../outputs/experiments/city_full_occlusion_30pct_20261025/review_predictions_class_mean_occlusion30_500.json)独立保存。用原说明中的命令启动 `scripts/real_review_server.py` 后，在浏览器选择新模型和新人工比较组；当前已有服务不会自动加载修改过的页面。

## 复现入口与检查范围

从 `mamma_retrain` 目录运行，使用已有 MAMMA 环境。计划已冻结；输出目录必须是新的，不能覆盖本轮结果。

```bash
export PYTHONPATH=. PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
MAMMA_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
CONFIG=configs/retrain/mamma_city_full_occlusion_30pct_20261025.json
RUN=outputs/experiments/city_full_occlusion_30pct_reproduction

# 仅在首次建立实验、sampling_plan.json 尚不存在时执行一次：
# CUDA_VISIBLE_DEVICES='' "$MAMMA_PY" scripts/freeze_occlusion30_plan.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES='' "$MAMMA_PY" scripts/mamma_retrain.py plan --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 "$MAMMA_PY" scripts/mamma_retrain.py train --config "$CONFIG" \
  --output "$RUN/train_25" --device cuda:0 --max-updates 25 --checkpoint-every 25 --execute
CUDA_VISIBLE_DEVICES=0 "$MAMMA_PY" scripts/mamma_retrain.py train --config "$CONFIG" \
  --output "$RUN/train_resume_500" --device cuda:0 --max-updates 500 --checkpoint-every 250 \
  --checkpoint "$RUN/train_25/checkpoint_update_000025.pt" --execute
CUDA_VISIBLE_DEVICES=0 "$MAMMA_PY" scripts/mamma_retrain.py evaluate --config "$CONFIG" \
  --split validation --output "$RUN/final_eval" --device cuda:0 \
  --checkpoint "$RUN/train_resume_500/checkpoint_update_000500.pt" --execute
```

定向 CPU 测试 13 项通过，覆盖计划配额、无放回至耗尽、确定性、恢复合同及现有损失／数据检查。页面使用隔离临时 Chrome 验证新模型切换、新组空白、评分导出再导入和旧组隔离；临时服务已关闭。没有渲染、扩批、真实 GT 评测、训练比例搜索或部署更新。
