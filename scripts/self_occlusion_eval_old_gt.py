"""Evaluate the trial checkpoint on the unchanged historical validation labels."""

import argparse
import json
from pathlib import Path

from mamma_ext.runner import main as runner_main


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retrain/mamma_self_occlusion_trial_20261025.json"
SPLIT = ROOT / "configs/retrain/city_merged_unified_validation_20261025_split.json"
EXPERIMENT = ROOT / "outputs/experiments/self_occlusion_unknown_trial_20261025"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("old_validation", "new_occlusion_validation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text())
    if split["schema"] != "mamma_data_split_v1" or len(split["validation"]) != 7:
        raise ValueError("historical_validation_split_changed")
    relative = split["validation"][:4] if args.group == "old_validation" else split["validation"][4:]
    command = ["evaluate", "--config", str(CONFIG), "--split", "validation",
               "--output", str(EXPERIMENT / "eval_old_gt" / args.group),
               "--device", args.device, "--checkpoint",
               str(EXPERIMENT / "train_full/checkpoint_update_000812.pt"), "--execute"]
    for item in relative:
        command.extend(("--eval-manifest", str((SPLIT.parent / item).resolve())))
    runner_main(command)


if __name__ == "__main__":
    main()
