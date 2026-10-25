"""Compare the four frozen models on the identical natural validation set."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.visibility_class_mean_compare import one_group
from scripts.visibility_supervision_stats import category


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/experiments/city_full_first_finetune_20261024"
BALANCED = ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024"
NEW = ROOT / "outputs/experiments/city_full_occlusion_30pct_20261025"
RUNS = {"original": OLD / "baseline_eval_retry1", "point_mean_500": OLD / "final_eval",
        "class_mean_500": BALANCED / "final_eval", "class_mean_occlusion30_500": NEW / "final_eval"}


def main():
    metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in RUNS.items()}
    files = {}
    for name, path in RUNS.items():
        with np.load(path / "predictions.npz", allow_pickle=False) as source:
            files[name] = {key: source[key].copy() for key in source.files}
    identities = [[row["identity"]["observation_id"] for row in metrics[name]["rows"]]
                  for name in RUNS]
    if len(identities[0]) != 435 or any(value != identities[0] for value in identities[1:]):
        raise ValueError("validation_observation_order_changed")
    for key in ("vertex_ids", "gt_erp_xy", "coordinate_loss_valid", "visibility"):
        if any(not np.array_equal(files[name][key], files["original"][key]) for name in RUNS):
            raise ValueError("validation_ground_truth_changed:" + key)
    groups = {"overall": list(range(435))}
    names = [category(row["identity"]["episode"]) for row in metrics["original"]["rows"]]
    for name in ("one_cycle", "two_cycle", "cycle_walk", "three_cycle"):
        groups[name] = [index for index, value in enumerate(names) if value == name]
    plan = json.loads((NEW / "sampling_plan.json").read_text())
    updates = [json.loads(line) for folder in ("train_25", "train_resume_500")
               for line in (NEW / folder / "updates.jsonl").read_text().splitlines()]
    if (len(updates) != 500 or [row["global_update"] for row in updates] != list(range(1, 501))
        or any(row["skip_optimizer"] or row["observation_id"] != plan["entries"][i]["observation_id"]
               or row["sampling_pool"] != plan["entries"][i]["pool"] for i, row in enumerate(updates))):
        raise ValueError("actual_updates_differ_from_frozen_sampling_plan")
    count_pool = Counter(row["sampling_pool"] for row in updates)
    if count_pool != {"occlusion": 150, "other": 350}:
        raise ValueError("actual_sampling_quota_changed")
    result = {
        "schema": "mamma_occlusion30_comparison_v1", "validation_samples": 435,
        "same_validation_identity_order_gt_and_template": True, "visibility_threshold": 0.5,
        "sampling_plan_id": plan["plan_id"],
        "training": {"effective_updates": 500, "pool_updates": dict(count_pool),
                     "distinct_observations": {pool: len({row["observation_id"] for row in updates
                                                         if row["sampling_pool"] == pool})
                                               for pool in count_pool},
                     "by_episode": dict(Counter(row["episode"] for row in updates)),
                     "by_category": dict(Counter(category(row["episode"]) for row in updates)),
                     "train_25_elapsed_seconds": json.loads((NEW / "train_25/completed.json").read_text())["elapsed_seconds"],
                     "resume_475_elapsed_seconds": json.loads((NEW / "train_resume_500/completed.json").read_text())["elapsed_seconds"]},
        "models": {name: {group: one_group(files[name], indices) for group, indices in groups.items()}
                   for name in RUNS},
    }
    output = NEW / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: result["models"][name]["overall"] for name in RUNS}, indent=2))


if __name__ == "__main__":
    main()
