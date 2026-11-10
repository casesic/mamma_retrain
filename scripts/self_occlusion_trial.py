"""Create one independent MAMMA bundle split with conservative self-occlusion negatives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = ROOT / "configs/retrain/city_merged_unified_validation_20261025_split.json"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/self_occlusion_unknown_trial_20261025"
POLICY = "mamma512_self_occlusion_evidence_4cm_trial_v1"
REASON_CODE = np.uint8(8)  # 0..7 belong to the original generator policy.
DEPTH_GAP_M = 0.04


def candidates(labels: dict[str, np.ndarray], observed_indices: set[int]) -> np.ndarray:
    """Require body first-hit and the same-person rendered foreground to agree."""
    old = labels["visibility_value"]
    if old.ndim != 2 or old.shape[1] != 512:
        raise ValueError("invalid_visibility_shape")
    observed = np.zeros((len(old), 1), dtype=bool)
    observed[list(observed_indices)] = True
    body = labels["body_first_range_m"]
    target = labels["gt_ray_range_m"]
    gap = labels["gt_minus_front_range_m"]
    xy = labels["xy_full_erp"]
    owner = labels["person_ids"][:, None]
    return (
        observed
        & (old == -1)
        & labels["geometry_defined"]
        & np.isfinite(xy).all(axis=2)
        & labels["physical_domain"]
        & labels["render_coverage"]
        & np.isfinite(target) & (target > 0)
        & np.isfinite(body) & (body > 0)
        & ((target - body) > DEPTH_GAP_M)
        & (labels["front_instance_id"] == owner)
        & np.isfinite(gap) & (gap > DEPTH_GAP_M)
        & (labels["nearer_same_front_id_3x3_count"] >= 5)
        & labels["evidence_neighborhood_complete"]
    )


def convert_labels(source: Path, destination: Path, observed_indices: set[int]) -> dict:
    with np.load(source, allow_pickle=False) as original:
        labels = {key: original[key].copy() for key in original.files}
    if str(labels["schema"]) != "mamma_512_label_v1":
        raise ValueError("unsupported_source_label_schema")
    before = labels["visibility_value"].copy()
    reason = labels["visibility_reason_code"].copy()
    selected = candidates(labels, observed_indices)
    labels["visibility_value"][selected] = 0
    labels["visibility_reason_code"][selected] = REASON_CODE
    labels["policy"] = np.asarray(POLICY)
    if not np.array_equal(labels["visibility_value"] != before, selected):
        raise ValueError("visibility_changed_outside_trial_rule")
    if not np.array_equal(labels["visibility_reason_code"][~selected], reason[~selected]):
        raise ValueError("original_reason_changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **labels)
    return {"changed": int(selected.sum()), "old_unknown_observed":
            int(np.count_nonzero(before[list(observed_indices)] == -1))}


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing_or_external_bundle_file:" + relative)
    return path


def _bundle(source_manifest: Path, output: Path) -> dict:
    source_root = source_manifest.parent
    bundle = json.loads(source_manifest.read_text())
    if bundle.get("schema") != "mamma_bundle_v1":
        raise ValueError("unsupported_bundle_schema")
    destination = output / "bundles" / bundle["episode"]
    if destination.exists():
        raise FileExistsError(destination)
    observed_by_label: dict[str, set[int]] = {}
    for row in bundle["observations"]:
        label_name = row["labels"]
        observed_by_label.setdefault(label_name, set())
        if row["status"] == "observed":
            observed_by_label[label_name].add(int(row["person_index"]))
    copied = set()
    totals = Counter()
    for row in bundle["observations"]:
        for field in ("rgb", "mask", "annotations"):
            relative = row[field]
            if relative in copied:
                continue
            source = _inside(source_root, relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)  # Only immutable raster/annotation inputs are linked.
            copied.add(relative)
        relative = row["labels"]
        if relative in copied:
            continue
        source = _inside(source_root, relative)
        counts = convert_labels(source, destination / relative, observed_by_label[relative])
        totals.update(counts)
        copied.add(relative)
    bundle["label_experiment"] = {
        "policy": POLICY, "source_manifest": str(source_manifest),
        "promoted_reason_code": int(REASON_CODE), "depth_gap_m": DEPTH_GAP_M,
        "changed_unknown_to_zero": totals["changed"],
    }
    (destination / "observations.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return {"episode": bundle["episode"], "source_manifest": str(source_manifest),
            "manifest": str(destination / "observations.json"),
            "observed": sum(row["status"] == "observed" for row in bundle["observations"]),
            "raw_unknown_observed": totals["old_unknown_observed"],
            "promoted_raw": totals["changed"]}


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(SOURCE_SPLIT.read_text())
    if original.get("schema") != "mamma_data_split_v1" or (len(original["train"]), len(original["validation"])) != (28, 7):
        raise ValueError("unexpected_frozen_split")
    output.mkdir(parents=True)
    split = {"schema": "mamma_data_split_v1", "train": [], "validation": []}
    records = []
    for group in ("train", "validation"):
        for relative in original[group]:
            source = (SOURCE_SPLIT.parent / relative).resolve()
            record = _bundle(source, output)
            record["split"] = group
            records.append(record)
            split[group].append(str(Path(record["manifest"]).relative_to(output)))
            print(group, record["episode"], record["promoted_raw"], flush=True)
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    summary = {"schema": "mamma_self_occlusion_trial_v1", "policy": POLICY,
               "source_split": str(SOURCE_SPLIT), "threshold_m": DEPTH_GAP_M,
               "reason_code": int(REASON_CODE), "bundles": records}
    (output / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
