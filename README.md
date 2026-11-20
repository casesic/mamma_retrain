<sub>[中文](#这是什么) · [English](#english)</sub>

> **这是 [`cuevhv/mamma`](https://github.com/cuevhv/mamma) 的 fork**（上游 MAMMA，CVPR 2026 Oral），不是官方仓库。
> 我加的是**微调流程 + 定量评价**：`mamma_ext/`、`scripts/mamma_retrain.py`、`configs/retrain/`。上游推理链没改（`landmarks/` 只动了一个读数据的文件）。权重与数据不在仓库里。

## 这是什么

用自产合成数据微调 MAMMA 的 2D 点位模型，并在统一验证集上量化：「比原模型好在哪、代价是什么」。

## 结果

阈值 0.5，统一验证集 7 场景 / 814 条观察：

| 指标 | 原模型 | 微调后 |
|---|---:|---:|
| ERP 点均误差 | 4.3954 px | **2.5242 px**（−42.6%） |
| 可见召回 | 82.64% | **98.42%** |
| 遮挡召回 | 95.73% | 85.94% |

**三条限制必须和数字一起读：**

1. **只在合成验证集上。** 真实骑行图没有人工点位真值，人工判断是整体画面意见，不是逐帧准确率。
2. **不是独立测试。** 模型是在这套划分上训练和报告的。
3. **遮挡召回掉了。** 95.73% → 85.94%，误判为可见的遮挡点 929 → 3059。点位误差和可见点的改善，是用遮挡换来的，不能只报前两行。

（0.85 阈值是另一个工作点：可见 91.33% / 遮挡 94.76%。它属于不同取舍，不是本文口径。）

## 怎么跑

前提：上游环境见 [docs/INSTALL.md](docs/INSTALL.md)；**数据自备**（`mamma_bundle_v1` 数据包，由 `human_data_generator` 产出）；权重与 `outputs/` 不进 Git。

```bash
export PYTHONPATH=.:src

# 只读检查：清单、包内文件、身份、点序；不加载模型、不碰 GPU
python scripts/mamma_retrain.py plan --config configs/retrain/mamma_regular.json --split validation

# 训练：3245 条有监督观察各一次，batch=4，812 次更新
python scripts/mamma_retrain.py train \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --device cuda:0 --max-updates 812 --checkpoint-every 500 \
  --output outputs/experiments/city_merged_one_epoch_20261025/train_full --execute

# 从保存的逐样本预测重算 814 条指标
python -m scripts.full_epoch_compare --combined
```

## 文档

| 想知道 | 去哪 |
|---|---|
| 现在用哪个权重、哪些说法已过期 | [docs/MAMMA_CURRENT_ENTRY.md](docs/MAMMA_CURRENT_ENTRY.md) |
| 17 篇实验报告哪篇还算数 | [docs/INDEX.md](docs/INDEX.md) |
| 数字怎么算出来的、代价是什么 | [docs/CITY_MERGED_FULL_EPOCH_20261025.md](docs/CITY_MERGED_FULL_EPOCH_20261025.md) |
| 交付边界与权重校验值 | [docs/DELIVERY_20261025.md](docs/DELIVERY_20261025.md) |
| 代码怎么读 | [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) |
| 四个仓库怎么交接 | [docs/PIPELINE_CONTRACT.md](docs/PIPELINE_CONTRACT.md) |
| 上游怎么用 | 下面的 Install / Quick demo / Pipeline / GUI |

## English
{: #english}

**Fork of [cuevhv/mamma](https://github.com/cuevhv/mamma)** (MAMMA, CVPR 2026 Oral), not the upstream repo. This fork adds a fine-tuning pipeline with a quantitative evaluation; first-party code lives in `mamma_ext/`, `scripts/mamma_retrain.py` and `configs/retrain/`.

Threshold 0.5, unified validation set (7 scenes / 814 observations):

| Metric | Original | Fine-tuned |
|---|---:|---:|
| Mean ERP landmark error | 4.3954 px | **2.5242 px** (−42.6%) |
| Visible recall | 82.64% | **98.42%** |
| Occluded recall | 95.73% | 85.94% |

Three caveats travel with these numbers: synthetic validation only (real footage has no human landmark ground truth); not an independent test; and **occluded recall falls from 95.73% to 85.94%** — the point-error gain is paid for with occluded points. Threshold 0.85 is a different operating point (91.33% / 94.76%) and is not the figure used here.

Weights, data and `outputs/` are not in the repo. Docs: [current state](docs/MAMMA_CURRENT_ENTRY.md), [report index](docs/INDEX.md), [pipeline contract](docs/PIPELINE_CONTRACT.md). Upstream usage (install, demo, datasets, GUI) is in [docs/UPSTREAM.md](docs/UPSTREAM.md).

## 上游怎么用

上游的安装、demo、数据集、GUI 说明在 [docs/UPSTREAM.md](docs/UPSTREAM.md)（原文照搬）。本 fork 不改变上游的推理链，装好环境后即可按那份说明跑 `python -m inference run`。
