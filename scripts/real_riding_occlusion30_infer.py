"""Infer the fixed 319 real targets with the 30% occlusion-sampling model."""

from pathlib import Path

import torch

from scripts.real_riding_class_mean_infer import run


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/city_full_occlusion_30pct_20261025"


if __name__ == "__main__":
    run(torch.device("cuda:0"),
        config_path=ROOT / "configs/retrain/mamma_city_full_occlusion_30pct_20261025.json",
        output=OUTPUT,
        checkpoint=OUTPUT / "train_resume_500/checkpoint_update_000500.pt",
        model_name="class_mean_occlusion30_500")
