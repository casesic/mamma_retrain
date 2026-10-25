"""Compare fixed models separately on the old and new natural validation sets."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.visibility_class_mean_compare import one_group


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/city_full_plus_occlusion_30pct_20261025"
OLD30 = ROOT / "outputs/experiments/city_full_occlusion_30pct_20261025"
ORIGINAL = ROOT / "outputs/experiments/city_full_first_finetune_20261024"
RUNS = {
    "old_validation": {
        "original": ORIGINAL / "baseline_eval_retry1",
        "old_data_30pct": OLD30 / "final_eval",
        "merged_data_30pct": OUTPUT / "old_validation_merged30",
    },
    "new_occlusion_validation": {
        "original": OUTPUT / "new_validation_original",
        "old_data_30pct": OUTPUT / "new_validation_old30",
        "merged_data_30pct": OUTPUT / "new_validation_merged30",
    },
}


def category(episode):
    if "two_cycle_straight" in episode:
        return "two_cycle_straight"
    if "two_cycle_diagonal" in episode:
        return "two_cycle_diagonal"
    if "three_cycle_straight" in episode:
        return "three_cycle_straight"
    if "one_cycle_one_walk" in episode:
        return "cycle_walk"
    for name in ("cycle_walk", "one_cycle", "two_cycle", "three_cycle"):
        if name in episode:
            return name
    raise ValueError("unknown_validation_episode:" + episode)


def compare_validation(runs, expected_samples):
    metrics = {name: json.loads((folder / "metrics.json").read_text()) for name, folder in runs.items()}
    arrays = {}
    for name, folder in runs.items():
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            arrays[name] = {key: saved[key].copy() for key in saved.files}
    ids = [[row["identity"]["observation_id"] for row in metrics[name]["rows"]]
           for name in runs]
    if len(ids[0]) != expected_samples or any(order != ids[0] for order in ids[1:]):
        raise ValueError("validation_observation_order_changed")
    for key in ("vertex_ids", "gt_erp_xy", "coordinate_loss_valid", "visibility"):
        if any(not np.array_equal(arrays[name][key], arrays["original"][key]) for name in runs):
            raise ValueError("validation_target_changed:" + key)
    names = [category(row["identity"]["episode"]) for row in metrics["original"]["rows"]]
    groups = {"overall": list(range(expected_samples))}
    groups.update({name: [i for i, value in enumerate(names) if value == name]
                   for name in sorted(set(names))})
    return {
        "samples": expected_samples,
        "same_order_template_gt_and_masks": True,
        "models": {name: {group: one_group(arrays[name], indices)
                           for group, indices in groups.items()} for name in runs},
    }


def main():
    plan = json.loads((OUTPUT / "sampling_plan.json").read_text())
    updates = [json.loads(line) for part in ("train_25", "train_resume_500")
               for line in (OUTPUT / part / "updates.jsonl").read_text().splitlines()]
    if (len(updates) != 500 or
        [row["global_update"] for row in updates] != list(range(1, 501)) or
        any(row["skip_optimizer"] or row["observation_id"] != plan["entries"][i]["observation_id"]
            or row["sampling_pool"] != plan["entries"][i]["pool"] for i, row in enumerate(updates))):
        raise ValueError("actual_updates_differ_from_frozen_plan")
    quotas = Counter(row["sampling_pool"] for row in updates)
    if quotas != {"occlusion": 150, "other": 350} or len({row["observation_id"] for row in updates}) != 500:
        raise ValueError("quota_or_unique_observation_mismatch")
    output = {
        "schema": "mamma_merged_occlusion30_comparison_v1",
        "visibility_threshold": 0.5,
        "sampling_plan_id": plan["plan_id"],
        "training": {"effective_updates": 500, "pool_updates": dict(quotas),
                     "unique_observations": 500,
                     "episodes": dict(Counter(row["episode"] for row in updates)),
                     "categories": dict(Counter(category(row["episode"]) for row in updates))},
        "validations": {"old_validation": compare_validation(RUNS["old_validation"], 435),
                        "new_occlusion_validation": compare_validation(RUNS["new_occlusion_validation"], 379)},
    }
    path = OUTPUT / "comparison.json"
    with path.open("x") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({name: {model: row["overall"] for model, row in section["models"].items()}
                      for name, section in output["validations"].items()}, indent=2))


if __name__ == "__main__":
    main()
