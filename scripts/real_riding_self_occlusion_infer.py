"""Infer the one-pass self-occlusion trial on the frozen 319 real targets."""

from pathlib import Path

import torch

from scripts.real_riding_class_mean_infer import run


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/self_occlusion_unknown_trial_20261025"


if __name__ == "__main__":
    run(torch.device("cuda:0"),
        config_path=ROOT / "configs/retrain/mamma_self_occlusion_trial_20261025.json",
        output=OUTPUT,
        checkpoint=OUTPUT / "train_full/checkpoint_update_000812.pt",
        model_name="self_occlusion_full_epoch")
