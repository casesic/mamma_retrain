<sub>**语言 / Language：** [中文](#本-fork-做了什么微调与评价) · [English](#what-this-fork-adds)</sub>

> **Fork 说明：本仓是 [cuevhv/mamma](https://github.com/cuevhv/mamma) 的 fork（上游 MAMMA，CVPR 2026 Oral），不是上游官方仓库。**
> 我在上游多相机动作捕捉推理链之上，加了**微调/重训流程、可见性阈值研究和数据适配**：`mamma_ext/`、`scripts/mamma_retrain.py`、`configs/retrain/`；上游 `landmarks/` 只动了一个读数据的文件（见下）。
> 结果：合成验证 7 场景 / 814 条观察，ERP 点均误差 **4.3954 px → 2.5242 px**（约 −42.6%），可见召回 82.64% → 91.33%（可见判定阈值 0.5 → 0.85）。
> 只想跑我加的部分：见下方[「本 fork 做了什么」](#本-fork-做了什么微调与评价)；只用上游推理链：直接看 [Install](#install)。
> 权重、训练数据和输出都不在仓库内（[交付说明](docs/DELIVERY_20261025.md)）；17 篇实验报告的状态索引在 [docs/INDEX.md](docs/INDEX.md)。

<div align="center">

# MAMMA: Markerless Accurate Multi-person Motion Acquisition

Hanz Cuevas Velasquez<sup>1\*</sup>, Anastasios Yiannakidis<sup>1\*</sup>, Soyong Shin<sup>2</sup>, Giorgio Becherini<sup>1</sup>, Markus Höschle<sup>1</sup>, Joachim Tesch<sup>1</sup>, Taylor Obersat<sup>1</sup>, Tsvetelina Alexiadis<sup>1</sup>, Eni Halilaj<sup>2</sup>, Michael J. Black<sup>1</sup>

<sup>1</sup>Max Planck Institute for Intelligent Systems, Tübingen &nbsp;&nbsp; <sup>2</sup>Carnegie Mellon University

<sup>\*</sup>Equal contribution

![teaser.png](docs/teaser.png)

**[CVPR 2026 Oral]** | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Velasquez_MAMMA_Markerless_Accurate_Multi-person_Motion_Acquisition_CVPR_2026_paper.html) | [arXiv](https://arxiv.org/abs/2506.13040) | [Project Page](https://mamma.is.tue.mpg.de/) | [Datasets](https://mamma.is.tue.mpg.de/download.php)

</div>

---

## 本 fork 做了什么（微调与评价）

上游 MAMMA 给的是「多相机视频 → SMPL-X」推理链（`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`）。**本 fork 在这条链之上，加了「用自产合成数据微调 2D 点位模型并做定量评价」这一层**，自写实现都在 `mamma_ext/`（10 个文件）。

与上游最后一次合并（`588492f`）相比，本 fork 的改动落在 `mamma_ext/`、`scripts/`、`configs/`、`tests/`、`review/`、`docs/`；`landmarks/` 里只有一个文件被动过：`landmarks/lib/datasets/vitdet_dataset.py`（+40 / −12），加了 `fast_inference` 分支，只对扩张后的人物框做局部裁剪和高斯模糊，其余逻辑不变。`mamma_ext/upstream.py` 只复用该目录 `utils_eval.py` 里的两个裁剪数学函数，`utils_eval.py` 本身未改。

| 我加的部分 | 在哪 | 做什么 |
|---|---|---|
| 数据 / 任务 / 模型 / 恢复适配 | `mamma_ext/` | 逐包读取 `mamma_bundle_v1` 数据包、由目标 mask 求框、有序 512 点标签、坐标 + uncertainty + visibility 损失、checkpoint 严格恢复 |
| 训练与评价入口 | `scripts/mamma_retrain.py` | `plan` / `check-data` / `train` / `evaluate` 四条子命令 |
| 训练配置与数据划分 | `configs/retrain/` | 通用配置 `execution_authorized=false`，不显式 `--execute` 不会训练 |
| 实验记录 | `docs/*_2026*.md` + [docs/INDEX.md](docs/INDEX.md) | 17 篇按日期归档，逐篇标注「结论仍有效 / 已被覆盖 / 仅历史」 |

完整状态与限制先看 [docs/MAMMA_CURRENT_ENTRY.md](docs/MAMMA_CURRENT_ENTRY.md)，交付边界看 [docs/DELIVERY_20261025.md](docs/DELIVERY_20261025.md)，改代码从 [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) 进。

### 结果

原模型（可见阈值 0.5）对比本轮微调权重（0.85），同一套统一验证集：

| 指标（合成验证：7 场景 / 814 条观察） | 原模型（阈值 0.5） | 微调后（阈值 0.85） |
|---|---:|---:|
| ERP 点均误差 | 4.3954 px | **2.5242 px** |
| 可见召回 | 82.64% | **91.33%** |
| 遮挡召回 | 95.73% | 94.76% |

点均误差下降约 **42.6%**。注意两栏阈值不同、不能混用：同一份微调权重在默认阈值 0.5 下是可见召回 98.42% / 遮挡召回 85.94%，换成 0.85 是拿可见漏判换回遮挡召回的结果。以上每个数字的原句都在 [docs/DELIVERY_20261025.md](docs/DELIVERY_20261025.md)，逐点重算过程在 [docs/CITY_MERGED_FULL_EPOCH_20261025.md](docs/CITY_MERGED_FULL_EPOCH_20261025.md)。

**引用这三组数字时必须同时带上三条 caveat**（原文见上述交付说明）：

1. **阈值是同集选的。** 0.85 是看过这份验证集之后选的，结果属于同集选择后的描述性结果，不是独立验证得出的最优点。
2. **不是独立测试。** 这是合成验证集；真实骑行图片没有人工点位真值，目前的人工判断是总体画面意见，不是逐帧评分或真实准确率。
3. **配对可能不同。** 40 对真实图片里共同成功的 35 帧，可用点对 2,474 → 3,665（约 +48.1%），但人物配对可能不同，不能据此宣称点对精度提高；更多点尚未转化成明显的合图视觉提升。

另外，微调权重目前**没有替换部署 engine**，演示合图仍是原模型 0.5 与本轮 0.85 并排对照。

### 复现我的部分

前提：上游环境按 [docs/INSTALL.md](docs/INSTALL.md) 装好；**训练与验证数据需自备**——每个包是 `mamma_bundle_v1` 结构（`observations.json` + RGB + 目标人物 mask + 有序 512 点标签），由同门的 `human_data_generator` 产出，本仓库不含数据、权重和 `outputs/`。所以这里能复现的是流程与命令，不是一串可直接下载的权重；没有这套数据时，第 1 步的合同检查也会直接报缺包。

```bash
export PYTHONPATH=.:src
export PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1

# 1) 只读合同检查（不加载模型、不碰 GPU）：清单 / 包内文件 / 身份 / 时间 / 点序
python scripts/mamma_retrain.py plan      --config configs/retrain/mamma_regular.json --split validation
python scripts/mamma_retrain.py check-data --config configs/retrain/mamma_regular.json --split validation --limit 2

# 2) 冻结本次取样顺序（一次完整遍历，已有计划不可重写）
python -m scripts.freeze_one_epoch_plan \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --output outputs/experiments/city_merged_one_epoch_20261025/sampling_plan.json

# 3) 训练：3245 条有监督观察各一次，batch=4，共 812 次优化器更新（尾批 1 条）
#    输出目录必须是新的；不加 --execute 只做前置检查
python scripts/mamma_retrain.py train \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --device cuda:0 --max-updates 812 --checkpoint-every 500 \
  --output outputs/experiments/city_merged_one_epoch_20261025/train_full --execute

# 4) 从已保存的逐样本预测统一重算 814 条指标（不重新加载模型或图片）
python -m scripts.full_epoch_compare --combined
```

预期产物：`outputs/experiments/city_merged_one_epoch_20261025/` 下的 checkpoint、`sampling_plan.json` 与 `combined_validation_comparison.json`。这些都在 `.gitignore` 里，不会进 Git。

---

## What this fork adds

This repo is a fork of [cuevhv/mamma](https://github.com/cuevhv/mamma) (MAMMA, CVPR 2026 Oral) — it is **not** the upstream repository. Upstream ships the multi-camera video → SMPL-X inference chain (`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`). **This fork adds the layer above that chain: fine-tuning the 2D landmark model on self-produced synthetic data, with a quantitative evaluation of the result.** All first-party code lives in `mamma_ext/` (10 files).

Measured against upstream's last merge (`588492f`), the changes touch `mamma_ext/`, `scripts/`, `configs/`, `tests/`, `review/` and `docs/`. Exactly one file under `landmarks/` was modified: `landmarks/lib/datasets/vitdet_dataset.py` (+40 / −12) gains a `fast_inference` branch that crops and blurs only the expanded person box; the rest of its logic is unchanged. `mamma_ext/upstream.py` reuses two crop-math helpers from `landmarks/lib/datasets/utils_eval.py`, which is untouched.

| Added | Where | What it does |
|---|---|---|
| Data / task / model / recovery adapters | `mamma_ext/` | Reads `mamma_bundle_v1` packages one by one, derives the box from the target mask, loads ordered 512-point labels, coordinate + uncertainty + visibility losses, strict checkpoint recovery |
| Training & evaluation CLI | `scripts/mamma_retrain.py` | `plan` / `check-data` / `train` / `evaluate` |
| Training configs & splits | `configs/retrain/` | The generic config is `execution_authorized=false`; nothing trains without an explicit `--execute` |
| Experiment records | `docs/*_2026*.md` + [docs/INDEX.md](docs/INDEX.md) | 17 dated reports, each labelled 结论仍有效 / 已被覆盖 / 仅历史 (still valid / superseded / history only) |

For current state and its limits start at [docs/MAMMA_CURRENT_ENTRY.md](docs/MAMMA_CURRENT_ENTRY.md), for the delivery boundary at [docs/DELIVERY_20261025.md](docs/DELIVERY_20261025.md), and to modify code at [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md).

### Results

Original model (visibility threshold 0.5) vs. this fork's fine-tuned weights (0.85), on one unified validation set:

| Metric (synthetic validation: 7 scenes / 814 observations) | Original (threshold 0.5) | Fine-tuned (threshold 0.85) |
|---|---:|---:|
| Mean ERP landmark error | 4.3954 px | **2.5242 px** |
| Visible recall | 82.64% | **91.33%** |
| Occluded recall | 95.73% | 94.76% |

Landmark error drops by about **42.6%**. The two column thresholds differ and must not be mixed: at the default 0.5 the same fine-tuned weights give 98.42% visible / 85.94% occluded recall, and 0.85 trades visible misses back for occluded recall. Every number above is quoted verbatim in [docs/DELIVERY_20261025.md](docs/DELIVERY_20261025.md); the point-by-point recomputation is in [docs/CITY_MERGED_FULL_EPOCH_20261025.md](docs/CITY_MERGED_FULL_EPOCH_20261025.md).

**Three caveats travel with these numbers** (wording from the delivery note):

1. **The threshold was chosen on the same set.** 0.85 was picked after looking at this validation set, so the result is a descriptive number from an in-set choice, not an independently validated optimum.
2. **Not an independent test.** The validation set is synthetic. Real riding footage has no human-labelled landmark ground truth, so the human judgment so far is a holistic impression of the stitched images, not per-frame scoring or real accuracy.
3. **Pairings may differ.** Of 40 real image pairs, 35 deformed successfully under both models; usable point pairs went 2,474 → 3,665 (≈ +48.1%). But the person pairings may differ, so this does not show higher point-pair accuracy, and the extra points have not yet turned into an obvious visual gain in the stitched images.

The fine-tuned weights have **not replaced the deployed engine**; the demo comparison still runs the original model at 0.5 side by side with this one at 0.85.

### Reproducing this fork's part

Prerequisites: the upstream environment per [docs/INSTALL.md](docs/INSTALL.md), and **your own training/validation data** — each package is a `mamma_bundle_v1` layout (`observations.json` + RGB + target-person mask + ordered 512-point labels) produced by the sibling `human_data_generator` repo. Weights, data and `outputs/` are not in this repo, so what reproduces here is the procedure, not a downloadable checkpoint; without that data, step 1 fails on missing packages.

```bash
export PYTHONPATH=.:src
export PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1

# 1) Read-only contract checks (no model, no GPU): manifest / package files / identity / timestamps / point order
python scripts/mamma_retrain.py plan      --config configs/retrain/mamma_regular.json --split validation
python scripts/mamma_retrain.py check-data --config configs/retrain/mamma_regular.json --split validation --limit 2

# 2) Freeze the sampling order for this run (one full pass; an existing plan is never rewritten)
python -m scripts.freeze_one_epoch_plan \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --output outputs/experiments/city_merged_one_epoch_20261025/sampling_plan.json

# 3) Train: each of the 3245 supervised observations once, batch=4, 812 optimizer updates (last batch has 1)
#    The output directory must be new; without --execute this is a preflight check only
python scripts/mamma_retrain.py train \
  --config configs/retrain/mamma_city_merged_one_epoch_20261025.json \
  --device cuda:0 --max-updates 812 --checkpoint-every 500 \
  --output outputs/experiments/city_merged_one_epoch_20261025/train_full --execute

# 4) Recompute the 814-observation metrics from saved per-sample predictions (no model or image reload)
python -m scripts.full_epoch_compare --combined
```

Expected artifacts: the checkpoint, `sampling_plan.json` and `combined_validation_comparison.json` under `outputs/experiments/city_merged_one_epoch_20261025/`. All of it is gitignored.

## News
- [2026-06] 🎉 MAMMA being presented at CVPR 2026
- [2026-06] Code released (inference + training)


---

## Install

```bash
git clone https://github.com/cuevhv/mamma.git
cd mamma
```

Full env + CUDA + weights setup: **[docs/INSTALL.md](docs/INSTALL.md)**.

```bash
micromamba activate mamma          # or: conda activate mamma
python -m inference doctor         # verify env vars + weight paths
```

The pipeline is zero-config when weights live under `data/`.

---

## Quick demo

Bundled 4-cam example, ~56 MB:

```bash
bash data/download_example.sh                                       # fetches videos to data/mamma_example/
```
```bash
python -m inference run \
  --cfg      configs/examples/presets/quick.yaml \
  --footage  data/mamma_example \
  --seq_name pushing_and_lifting_from_ground \
  --calib    configs/examples/calib/iphones_outdoors.yaml \
  --out-tag  demo -v
```

Outputs land under `output/ma_*/demo/mamma_example/…`.

**Prefer a browser UI?** Run `bash gui/scripts/dev.sh`, open <http://localhost:3000>, and click **Run demo**. It's the same pipeline but friendlier UX!

---

## Pipeline

`ma_cap → ma_masks → ma_2d → ma_3d → ma_vis`

| Step       | What it does                                 |
|------------|----------------------------------------------|
| `ma_cap`   | Loads multi-view capture |
| `ma_masks` | Per-person segmentation (SAM + YOLO)         |
| `ma_2d`    | 2D landmark detection (MammaNet)             |
| `ma_3d`    | Multi-view SMPL-X optimization               |
| `ma_vis`   | Per-camera overlays + interactive scene      |

Entry point: `python -m inference run` (source: [`inference/cli/run.py`](inference/cli/run.py)).

| Argument               | What it is                                                                                  |
|------------------------|---------------------------------------------------------------------------------------------|
| `--cfg` / `--preset`   | Pipeline-configuration YAML — declares which steps run and their hyperparameters. Capture-independent. ([what a preset is + how to modify one](docs/CONFIGS.md)) |
| `--footage`            | Dataset root containing sequence subdirs (use with `--seq_name` + `--calib`). ([layout reference](docs/YOUR-DATA.md#1-lay-out-your-footage)) |
| `--seq_name`           | One sequence subdirectory name under `--footage` to process (one run = one sequence).      |
| `--calib`              | Calibration file (`.yaml` / `.xcp` / OpenCV `.json`); applies to every sequence under `--footage`. ([format reference](docs/YOUR-DATA.md#2-author-the-calibration-file)) |
| `--capture`            | Advanced: capture JSON pointing at footage, calibration, sequences, and camera names — used to iterate over many sequences in one invocation. ([schema reference](docs/YOUR-DATA.md#3-mint-a-capture-descriptor)) |
| `--out-tag`            | Output sub-directory tag under `output/ma_*/<tag>/` (default: `local`).                     |
| `-v`                   | Verbose runner logs.                                                                        |

### Run the pipeline

Three things are needed:

1. A **calibration file** ([how to make one](docs/YOUR-DATA.md#2-author-the-calibration-file))
2. A **folder with your sequence** ([how to set it up](docs/YOUR-DATA.md#1-lay-out-your-footage))
3. A **preset** — use a shipped one: [`configs/examples/presets/quick.yaml`](configs/examples/presets/quick.yaml) (~5 min smoke) or [`configs/examples/presets/full.yaml`](configs/examples/presets/full.yaml) (full-frame). See [`docs/CONFIGS.md`](docs/CONFIGS.md) to modify or author your own.

Then:

```bash
python -m inference run \
  --cfg      <path/to/preset>.yaml \
  --footage  <path/to/footage> \
  --seq_name <seq_name> \
  --calib    <path/to/calib>.yaml \
  --out-tag  run01 -v
```

**Alternative — iterate over many sequences in one invocation.** A [capture JSON](docs/YOUR-DATA.md#3-mint-a-capture-descriptor) enumerates sequences, cameras, and the calibration in one file; the runner walks them automatically:

```bash
python -m inference run \
  --cfg     <path/to/preset>.yaml \
  --capture <path/to/capture>.json \
  --out-tag run01 -v
```

---

## GUI

Browser UI for submitting and inspecting runs. It uses the same `mamma` python env.

```bash
gui/scripts/dev.sh        # dev: Flask :8000 + Vite :3000 (auto-reload)
gui/scripts/prod.sh       # prod: single Flask process on :8000
```

Setup and deployment: [`gui/README.md`](gui/README.md).

---

## MAMMA datasets

The paper's released captures, evaluation data, and synthetic training data live on the MAMMA project page and require a free account.

1. Register at <https://mamma.is.tue.mpg.de/> and confirm your email.
2. Either use the GUI's *Pipeline assets* panel (sign in once, click to download), or run the per-dataset shell scripts under [`data/`](data/):

   ```bash
   bash data/download_mamma_dance.sh --bachata --meta --pred --videos_crf24
   ```

Five dataset families ship: **dance**, **multi-person**, **iPhone**, **eval**, and **synthetic**. Per-dataset sizes, video encodings, and the full script flag surface live in **[docs/DATASETS.md](docs/DATASETS.md)**.

> Just running on your own footage? You don't need any of this — see [Run the pipeline](#run-the-pipeline) above.

---

## Layout

```
.
├── inference/       runner, step builders, doctor CLI
├── capture/         ma_cap step
├── segmentation/    ma_masks step
├── landmarks/       ma_2d step
├── optimization/    ma_3d step
├── visualization/   ma_vis step
├── configs/         presets + capture manifests
├── data/            body models + weights + datasets (gitignored)
├── output/          run outputs (gitignored)
├── gui/             browser UI (Flask + React)
└── scripts/         smoke tests + utilities
```

---

## TODO

- [ ] Release the evaluation scripts (2D landmark + benchmark evaluation) and the processed evaluation datasets.

---

## Citation

```
@inproceedings{cuevas2026mamma,
  title     = {{MAMMA}: {Markerless Accurate Multi-person Motion Acquisition}},
  author    = {Cuevas Velasquez, Hanz and Yiannakidis, Anastasios and Shin, Soyong and Becherini, Giorgio and H{\"o}schle, Markus and Tesch, Joachim and Obersat, Taylor and Alexiadis, Tsvetelina and Halilaj, Eni and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgments

MAMMA builds on a number of open-source models, datasets, and tools. We thank their authors for releasing their work openly.

- [**SAM 2**](https://github.com/facebookresearch/sam2) and [**SAM 3**](https://github.com/facebookresearch/sam3): Meta FAIR; segmentation backbones.
- [**YOLOv12**](https://github.com/ultralytics/ultralytics): Ultralytics; person detection.
- [**SMPL-X**](https://smpl-x.is.tue.mpg.de/): MPI-IS; expressive body model.
- [**ViTPose**](https://github.com/ViTAE-Transformer/ViTPose): pretrained human-pose backbone.
- [**HRNet**](https://github.com/HRNet): alternative pose-estimation backbone.
- [**CameraHMR**](https://camerahmr.is.tue.mpg.de/): MPI-IS; landmark architecture baseline.
- [**BEDLAM**](https://bedlam.is.tue.mpg.de/): MPI-IS; synthetic dataset of humans in motion.
- [**Rerun**](https://rerun.io): interactive 3D scene viewer.
- [**Detectron2**](https://github.com/facebookresearch/detectron2): Meta FAIR; vision research library.
- [**PyTorch Lightning**](https://github.com/Lightning-AI/pytorch-lightning), [**Hydra**](https://github.com/facebookresearch/hydra), and [**WebDataset**](https://github.com/webdataset/webdataset): training infrastructure.

## License

For non-commercial scientific research purposes [LICENSE](LICENSE).

## Contact

Questions, bug reports, or other inquiries: [mamma@tue.mpg.de](mailto:mamma@tue.mpg.de).
