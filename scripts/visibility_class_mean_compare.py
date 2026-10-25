"""Compare three fixed model outputs on the identical 435-observation validation set."""

import json
from pathlib import Path

import numpy as np

from scripts.visibility_supervision_stats import category


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/experiments/city_full_first_finetune_20261024"
NEW = ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024"
RUNS = {"original": OLD / "baseline_eval_retry1", "point_mean_500": OLD / "final_eval",
        "class_mean_500": NEW / "final_eval"}


def one_group(data, indices):
    predicted = data["predicted_erp_xy"][indices]
    target = data["gt_erp_xy"][indices]
    coordinate_valid = data["coordinate_loss_valid"][indices].astype(bool)
    labels = data["visibility"][indices]
    logits = data["visibility_logits"][indices]
    dx = (predicted[..., 0] - target[..., 0] + 2880) % 5760 - 2880
    distance = np.hypot(dx, predicted[..., 1] - target[..., 1])
    point_errors = distance[coordinate_valid]
    sample_means = [distance[i][coordinate_valid[i]].mean()
                    for i in range(len(indices)) if coordinate_valid[i].any()]
    by_label = {}
    for label, name in ((1, "visible"), (0, "occluded"), (-1, "unknown")):
        values = distance[coordinate_valid & (labels == label)]
        by_label[name] = {"points": int(values.size),
                          "mean_erp_px": float(values.mean()) if values.size else None}
    known = coordinate_valid & np.isin(labels, [0, 1])
    pos = known & (labels == 1)
    neg = known & (labels == 0)
    correct_pos = int(np.count_nonzero((logits > 0) & pos))
    incorrect_pos = int(np.count_nonzero((logits <= 0) & pos))
    correct_neg = int(np.count_nonzero((logits <= 0) & neg))
    incorrect_neg = int(np.count_nonzero((logits > 0) & neg))
    return {"samples": len(indices), "coordinate_points": int(point_errors.size),
            "point_mean_erp_px": float(point_errors.mean()) if point_errors.size else None,
            "sample_mean_erp_px": float(np.mean(sample_means)) if sample_means else None,
            "coordinate_by_visibility": by_label,
            "visibility": {"positive": int(pos.sum()), "negative": int(neg.sum()),
                           "true_visible": correct_pos, "visible_missed": incorrect_pos,
                           "true_occluded": correct_neg, "occluded_predicted_visible": incorrect_neg,
                           "visible_recall": correct_pos / int(pos.sum()) if pos.any() else None,
                           "occluded_recall": correct_neg / int(neg.sum()) if neg.any() else None}}


def main():
    metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in RUNS.items()}
    files = {}
    for name, path in RUNS.items():
        with np.load(path / "predictions.npz", allow_pickle=False) as npz:
            files[name] = {key: npz[key].copy() for key in npz.files}
    identities = [[row["identity"]["observation_id"] for row in metrics[name]["rows"]]
                  for name in RUNS]
    if len(identities[0]) != 435 or any(ids != identities[0] for ids in identities[1:]):
        raise ValueError("three_evaluations_have_different_observations")
    for key in ("vertex_ids", "gt_erp_xy", "coordinate_loss_valid", "visibility"):
        if any(not np.array_equal(files[name][key], files["original"][key]) for name in RUNS):
            raise ValueError(f"three_evaluations_have_different_ground_truth:{key}")
    names = [category(row["identity"]["episode"]) for row in metrics["original"]["rows"]]
    groups = {"overall": list(range(435))}
    groups.update({name: [i for i, value in enumerate(names) if value == name]
                   for name in ("one_cycle", "two_cycle", "cycle_walk", "three_cycle")})
    def updates(root):
        return [json.loads(line) for part in ("train_25", "train_resume_500")
                for line in (root / part / "updates.jsonl").read_text().splitlines()]

    old_updates, new_updates = updates(OLD), updates(NEW)
    if (len(old_updates) != len(new_updates) or len(new_updates) != 500 or
        [row["observation_id"] for row in old_updates] !=
        [row["observation_id"] for row in new_updates] or
        [row["global_update"] for row in new_updates] != list(range(1, 501))):
        raise ValueError("training_update_order_or_count_changed")
    by_episode = {}
    for row in new_updates:
        by_episode[row["episode"]] = by_episode.get(row["episode"], 0) + 1
    source_weight = ROOT / "outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt"
    result = {"schema": "visibility_class_mean_comparison_v1",
              "same_observations_ground_truth_and_template": True,
              "same_500_training_observation_order": True,
              "threshold": 0.5,
              "training": {"effective_updates": 500, "skipped": sum(row["skip_optimizer"] for row in new_updates),
                           "by_episode": by_episode,
                           "by_category": {name: sum(count for episode, count in by_episode.items()
                                                    if category(episode) == name)
                                           for name in ("one_cycle", "two_cycle", "cycle_walk", "three_cycle")},
                           "visibility_loss_first_25_mean": float(np.mean([row["visibility_loss"] for row in new_updates[:25]])),
                           "visibility_loss_last_25_mean": float(np.mean([row["visibility_loss"] for row in new_updates[-25:]])),
                           "source_weight": {"path": str(source_weight), "size_bytes": source_weight.stat().st_size,
                                             "mtime_ns": source_weight.stat().st_mtime_ns},
                           "gpu_uuid": "GPU-6f046d56-acbe-d49b-774a-9d463e768cac",
                           "train_25_elapsed_seconds": json.loads((NEW / "train_25/completed.json").read_text())["elapsed_seconds"],
                           "resume_475_elapsed_seconds": json.loads((NEW / "train_resume_500/completed.json").read_text())["elapsed_seconds"]},
              "models": {name: {group: one_group(files[name], indices)
                                for group, indices in groups.items()} for name in RUNS}}
    (NEW / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result["models"][name]["overall"] for name in RUNS}, indent=2))


if __name__ == "__main__":
    main()
