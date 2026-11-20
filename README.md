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

Weights, data and `outputs/` are not in the repo. Docs: [current state](docs/MAMMA_CURRENT_ENTRY.md), [report index](docs/INDEX.md), [pipeline contract](docs/PIPELINE_CONTRACT.md). Upstream usage follows.

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
