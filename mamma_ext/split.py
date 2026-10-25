"""Keep whole episodes on one side of the MAMMA train/validation split."""

from collections import Counter


def observed_rows(dataset):
    return [dataset.slots[i] for i in dataset.ready]


def group_inventory(dataset):
    rows = observed_rows(dataset)
    absent = [row for row in dataset.slots if row["status"] == "absent"]
    return {
        "episodes": sorted({r["episode"] for r in dataset.slots}),
        "source_assets": sorted({r["source_asset_id"] for r in rows}),
        "source_manifests": sorted({r["source_manifest_id"] for r in rows}),
        "observed_target_count": len(rows),
        "absent_count": len(absent),
        "absent_reasons": dict(Counter(row["absent_reason"] for row in absent)),
    }


def validate_disjoint(train, validation):
    """A person row, neighboring frame or alternate rig cannot cross the split."""
    a, b = group_inventory(train), group_inventory(validation)
    overlap = sorted(set(a["episodes"]) & set(b["episodes"]))
    if overlap:
        raise ValueError("train_validation_group_overlap_episodes:" + ",".join(overlap))
    if hasattr(train, "cfg") and hasattr(validation, "cfg"):
        if set(train.cfg["manifests"]) & set(validation.cfg["manifests"]):
            raise ValueError("train_validation_duplicate_data_package")
    if not a["observed_target_count"] or not b["observed_target_count"]:
        raise ValueError("empty_train_or_validation_targets")
    return {"train": a, "validation": b, "grouping": "episode_v1"}
