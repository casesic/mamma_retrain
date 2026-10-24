"""Conservative grouping for MAMMA train/validation manifests."""


def observed_rows(dataset):
    return [dataset.slots[i] for i in dataset.ready]


def group_inventory(dataset):
    rows = observed_rows(dataset)
    return {
        "episodes": sorted({r["episode"] for r in rows}),
        "source_assets": sorted({r["source_asset_id"] for r in rows}),
        "source_manifests": sorted({r["source_manifest_id"] for r in rows}),
        "observed_target_count": len(rows),
        "absent_slots": [
            dataset.slot_status(i) for i, s in enumerate(dataset.slots) if s["status"] != "observed"
        ],
    }


def validate_disjoint(train, validation):
    """A person row, neighboring frame or alternate rig cannot cross the split."""
    a, b = group_inventory(train), group_inventory(validation)
    for key in ("episodes", "source_assets", "source_manifests"):
        overlap = sorted(set(a[key]) & set(b[key]))
        if overlap:
            raise ValueError("train_validation_group_overlap_" + key + ":" + ",".join(overlap))
    if not a["observed_target_count"] or not b["observed_target_count"]:
        raise ValueError("empty_train_or_validation_targets")
    return {"train": a, "validation": b, "grouping": "episode_and_source_asset_manifest_v1"}
