"""Infer the merged-data full-epoch model on the frozen real-riding targets."""

from pathlib import Path

import torch

from scripts.real_riding_class_mean_infer import run


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/city_merged_one_epoch_20261025"


if __name__ == "__main__":
    run(torch.device("cuda:0"),
        config_path=ROOT / "configs/retrain/mamma_city_merged_one_epoch_20261025.json",
        output=OUTPUT,
        checkpoint=OUTPUT / "train_full/checkpoint_update_000812.pt",
        model_name="class_mean_merged_full_epoch")
