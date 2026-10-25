"""Verify one-pass coverage and compare original, full-epoch, and 500-step models."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.merged_occlusion30_compare import compare_validation, category
from scripts.visibility_class_mean_compare import one_group


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/experiments/city_merged_one_epoch_20261025"
PRIOR = ROOT / "outputs/experiments/city_full_plus_occlusion_30pct_20261025"
ORIGINAL = ROOT / "outputs/experiments/city_full_first_finetune_20261024"
RUNS = {
    "old_validation": {
        "original": ORIGINAL / "baseline_eval_retry1",
        "full_epoch": OUTPUT / "old_validation_full_epoch",
        "merged_500": PRIOR / "old_validation_merged30",
    },
    "new_occlusion_validation": {
        "original": PRIOR / "new_validation_original",
        "full_epoch": OUTPUT / "new_validation_full_epoch",
        "merged_500": PRIOR / "new_validation_merged30",
    },
}


def combine_validation():
    """Recompute metrics on one 814-observation set from saved predictions."""
    from mamma_ext.data import MammaSavedGTDataset

    rows, manifests = [], []
    arrays = {model: [] for model in RUNS["old_validation"]}
    all_ids = []
    vertex_ids = None
    for group, runs in RUNS.items():
        expected = 435 if group == "old_validation" else 379
        compare_validation(runs, expected)
        group_metrics = {model: json.loads((folder / "metrics.json").read_text())
                         for model, folder in runs.items()}
        reference = group_metrics["full_epoch"]
        group_manifests = reference["evaluation_manifests"]
        if (not isinstance(group_manifests, list) or not group_manifests
            or any(metric.get("evaluation_manifests") not in (None, group_manifests)
                   for metric in group_metrics.values())
            or any(metric["crop_policy_version"] != "mask_bbox_periodic_v2"
                   or metric["gt_policy"] != "saved_mamma_512_npz_v1"
                   for metric in group_metrics.values())):
            raise ValueError("combined_validation_source_contract_mismatch")
        manifests.extend(group_manifests)
        rows.extend(reference["rows"])
        all_ids.extend(row["identity"]["observation_id"] for row in reference["rows"])
        for model, folder in runs.items():
            with np.load(folder / "predictions.npz", allow_pickle=False) as source:
                copied = {key: source[key].copy() for key in source.files}
            if vertex_ids is None:
                vertex_ids = copied["vertex_ids"]
            elif not np.array_equal(vertex_ids, copied["vertex_ids"]):
                raise ValueError("combined_validation_template_order_changed")
            arrays[model].append(copied)
    if len(manifests) != 7 or len(set(manifests)) != 7 or len(all_ids) != 814 or len(set(all_ids)) != 814:
        raise ValueError("combined_validation_duplicate_or_missing_source")
    split_file = ROOT / "configs/retrain/city_merged_unified_validation_20261025_split.json"
    split = json.loads(split_file.read_text())
    expected_manifests = [str((split_file.parent / name).resolve()) for name in split["validation"]]
    train_manifests = [str((split_file.parent / name).resolve()) for name in split["train"]]
    if (split.get("schema") != "mamma_data_split_v1" or expected_manifests != manifests
        or len(train_manifests) != 28 or set(train_manifests) & set(manifests)):
        raise ValueError("combined_validation_split_changed")
    dataset = MammaSavedGTDataset(manifests)
    reader_ids = [dataset.slots[slot]["observation_id"] for slot in dataset.ready]
    if reader_ids != all_ids:
        raise ValueError("combined_validation_reader_order_changed")
    combined = {
        model: {key: np.concatenate([part[key] for part in parts])
                for key in ("predicted_erp_xy", "gt_erp_xy", "coordinate_loss_valid",
                            "visibility", "visibility_logits")}
        for model, parts in arrays.items()
    }
    for key in ("gt_erp_xy", "coordinate_loss_valid", "visibility"):
        if any(not np.array_equal(data[key], combined["original"][key])
               for data in combined.values()):
            raise ValueError("combined_validation_target_or_mask_changed:" + key)
    labels = [category(row["identity"]["episode"]) for row in rows]
    groups = {"overall": list(range(814))}
    groups.update({name: [index for index, value in enumerate(labels) if value == name]
                   for name in sorted(set(labels))})
    result = {
        "schema": "mamma_combined_validation_comparison_v1",
        "primary_comparison": ["original", "full_epoch"],
        "validation_manifests": manifests,
        "validation_split": str(split_file),
        "observed_targets": len(all_ids),
        "episodes": sorted({row["identity"]["episode"] for row in rows}),
        "same_order_template_gt_and_masks": True,
        "source_evaluations": {group: {model: str(folder) for model, folder in runs.items()}
                               for group, runs in RUNS.items()},
        "models": {model: {name: one_group(data, indices) for name, indices in groups.items()}
                   for model, data in combined.items()},
    }
    destination = OUTPUT / "combined_validation_comparison.json"
    with destination.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({model: scores["overall"] for model, scores in result["models"].items()},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", action="store_true", help="one 814-observation validation result")
    args = parser.parse_args()
    if args.combined:
        combine_validation()
        return
    plan = json.loads((OUTPUT / "sampling_plan.json").read_text())
    log = [json.loads(line) for line in (OUTPUT / "train_full/updates.jsonl").read_text().splitlines()]
    entries = plan["entries"]
    actual_ids = [identity for row in log for identity in row["observation_ids"]]
    expected_ids = [row["observation_id"] for row in entries]
    if (actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids)
        or len(log) != (len(entries) + 3) // 4
        or [row["global_update"] for row in log] != list(range(1, len(log) + 1))
        or [row["batch_size"] for row in log] != [4] * (len(log) - 1) + [len(entries) % 4 or 4]
        or log[-1]["sample_position_after"] != len(entries)):
        raise ValueError("full_epoch_update_log_does_not_match_frozen_plan")
    result = {
        "schema": "mamma_full_epoch_comparison_v1",
        "primary_comparison": ["original", "full_epoch"],
        "batch_size": 4,
        "sampling_plan_id": plan["plan_id"],
        "training": {
            "trainable_observations": len(entries),
            "excluded_no_supervision": len(plan["excluded_no_supervision"]),
            "optimizer_updates": len(log),
            "unique_observations": len(set(actual_ids)),
            "occluded_observations": sum(row["negative"] > 0 for row in entries),
            "other_observations": sum(row["negative"] == 0 for row in entries),
            "episodes": dict(Counter(row["episode"] for row in entries)),
            "categories": dict(Counter(category(row["episode"]) for row in entries)),
        },
        "validations": {
            "old_validation": compare_validation(RUNS["old_validation"], 435),
            "new_occlusion_validation": compare_validation(RUNS["new_occlusion_validation"], 379),
        },
    }
    destination = OUTPUT / "comparison.json"
    with destination.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({section: {model: scores["overall"] for model, scores in data["models"].items()}
                      for section, data in result["validations"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
