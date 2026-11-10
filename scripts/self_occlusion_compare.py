"""Compare three fixed models on old GT and the independent new self-occlusion subset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mamma_ext.data import MammaSavedGTDataset
from scripts.full_epoch_compare import RUNS
from scripts.merged_occlusion30_compare import category
from scripts.visibility_class_mean_compare import one_group


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "outputs/experiments/self_occlusion_unknown_trial_20261025"
SOURCES = {
    "old_validation": {
        "original": RUNS["old_validation"]["original"],
        "old_full_epoch": RUNS["old_validation"]["full_epoch"],
        "trial_full_epoch": TRIAL / "eval_old_gt",
    },
    "new_occlusion_validation": {
        "original": RUNS["new_occlusion_validation"]["original"],
        "old_full_epoch": RUNS["new_occlusion_validation"]["full_epoch"],
        "trial_full_epoch": TRIAL / "eval_old_gt",
    },
}


def _read(path: Path):
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _new_negative_subset(old_ids: list[str], valid: np.ndarray) -> np.ndarray:
    split = json.loads((TRIAL / "split.json").read_text())
    dataset = MammaSavedGTDataset([str((TRIAL / name).resolve()) for name in split["validation"]])
    ids = [dataset.slots[slot]["observation_id"] for slot in dataset.ready]
    if ids != old_ids or not np.array_equal(dataset.vertex_ids.shape, (512,)):
        raise ValueError("trial_validation_order_or_template_changed")
    selected = []
    for slot in dataset.ready:
        row, files = dataset.slots[slot], dataset._files[slot]
        with np.load(files["labels"], allow_pickle=False) as labels:
            j = row["person_index"]
            selected.append(labels["visibility_reason_code"][j] == 8)
            if not np.all(labels["visibility_value"][j][selected[-1]] == 0):
                raise ValueError("trial_reason_without_negative")
    selected = np.stack(selected)
    if selected.shape != valid.shape or not np.all(valid[selected]) or selected.sum() != 156179:
        raise ValueError("trial_negative_mask_or_reader_support_changed")
    return selected


def compare() -> dict:
    files, ids, manifests = {}, [], []
    for group in ("old_validation", "new_occlusion_validation"):
        paths = SOURCES[group]
        part = {}
        reference = json.loads((paths["old_full_epoch"] / "metrics.json").read_text())
        group_ids = [row["identity"]["observation_id"] for row in reference["rows"]]
        ids.extend(group_ids)
        manifests.extend(reference["evaluation_manifests"])
        for name, path in paths.items():
            directory = path / group if name == "trial_full_epoch" else path
            metrics = json.loads((directory / "metrics.json").read_text())
            if ([row["identity"]["observation_id"] for row in metrics["rows"]] != group_ids
                    or metrics.get("evaluation_manifests", reference["evaluation_manifests"])
                    != reference["evaluation_manifests"]):
                raise ValueError("evaluation_identity_or_data_changed:" + name)
            part[name] = _read(directory)
        for key in ("vertex_ids", "gt_erp_xy", "coordinate_loss_valid", "visibility"):
            if any(not np.array_equal(part[name][key], part["original"][key]) for name in part):
                raise ValueError("old_gt_or_template_changed:" + key)
        files[group] = part
    if len(ids) != 814 or len(set(ids)) != 814 or len(manifests) != 7:
        raise ValueError("validation_coverage_changed")
    combined = {name: {key: np.concatenate([files[group][name][key]
                                               for group in files])
                       for key in ("vertex_ids", "predicted_erp_xy", "gt_erp_xy",
                                   "coordinate_loss_valid", "visibility", "visibility_logits")
                       if key != "vertex_ids"}
                for name in ("original", "old_full_epoch", "trial_full_epoch")}
    for name in combined:
        combined[name]["vertex_ids"] = files["old_validation"][name]["vertex_ids"]
    selected = _new_negative_subset(ids, combined["original"]["coordinate_loss_valid"])
    if not np.all(combined["original"]["visibility"][selected] == -1):
        raise ValueError("trial_subset_was_not_old_unknown")
    split = json.loads((TRIAL / "split.json").read_text())
    new_manifests = [str((TRIAL / name).resolve()) for name in split["validation"]]
    if len(new_manifests) != len(manifests):
        raise ValueError("trial_validation_package_count_changed")
    groups = {"overall": list(range(814))}
    names = [category(row["identity"]["episode"])
             for group in ("old_validation", "new_occlusion_validation")
             for row in json.loads((SOURCES[group]["original"] / "metrics.json").read_text())["rows"]]
    groups.update({name: [i for i, value in enumerate(names) if value == name]
                   for name in sorted(set(names))})
    result = {"schema": "mamma_self_occlusion_trial_comparison_v1", "threshold": 0.5,
              "old_validation_manifests": manifests,
              "trial_validation_manifests": new_manifests,
              "old_gt_samples": len(ids), "new_self_occlusion_points": int(selected.sum()),
              "models": {}}
    for name, data in combined.items():
        logits = data["visibility_logits"]
        predicted = data["predicted_erp_xy"]
        target = data["gt_erp_xy"]
        dx = (predicted[..., 0] - target[..., 0] + 2880) % 5760 - 2880
        error = np.hypot(dx, predicted[..., 1] - target[..., 1])
        result["models"][name] = {
            "old_gt": {group: one_group(data, indices) for group, indices in groups.items()},
            "new_self_occlusion_subset": {
                "points": int(selected.sum()),
                "predicted_occluded": int(np.count_nonzero((logits <= 0) & selected)),
                "predicted_visible": int(np.count_nonzero((logits > 0) & selected)),
                "occluded_recall": float(np.mean(logits[selected] <= 0)),
                "coordinate_mean_erp_px": float(error[selected].mean()),
            },
        }
    (TRIAL / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = compare()
    print(json.dumps({name: {"old_gt": section["old_gt"]["overall"],
                             "new_subset": section["new_self_occlusion_subset"]}
                      for name, section in result["models"].items()}, indent=2))
