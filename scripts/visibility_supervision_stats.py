"""Count the visibility labels that actually enter the fixed training loss, without RGB decoding."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mamma_ext.data import _targets, gen_trans_from_patch_cv, native_plan
from mamma_ext.policy import load_active_policy
from mamma_ext.runner import mixed_order


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retrain/mamma_city_full_visibility_class_mean_20261024.json"
OLD = ROOT / "outputs/experiments/city_full_first_finetune_20261024"
OUTPUT = ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024"


def category(episode):
    for name in ("cycle_walk", "one_cycle", "two_cycle", "three_cycle"):
        if name in episode or (name == "cycle_walk" and "one_cycle_one_walk" in episode):
            return name
    raise ValueError(f"unknown_episode_category:{episode}")


def summary():
    config, dataset = load_active_policy(CONFIG)
    previous = []
    for folder in ("train_25", "train_resume_500"):
        with (OLD / folder / "updates.jsonl").open() as source:
            previous.extend(json.loads(line) for line in source)
    if len(previous) != 500 or [row["global_update"] for row in previous] != list(range(1, 501)):
        raise ValueError("previous_500_updates_incomplete")
    order = mixed_order(dataset, config["sampler_seed"], 0)[:500]
    if [dataset.slots[dataset.ready[i]]["observation_id"] for i in order] != [
        row["observation_id"] for row in previous
    ]:
        raise ValueError("sampler_order_changed_from_point_mean_run")
    selected = set(order)
    totals = defaultdict(lambda: {"samples": 0, "positive": 0, "negative": 0,
                                  "both": 0, "positive_only": 0, "negative_only": 0, "none": 0})
    for index, slot in enumerate(dataset.ready):
        row = dataset.slots[slot]
        files = dataset._files[slot]
        support = row["crop_support"]
        box = np.asarray(support["box_xyxy"], np.int64)
        roi = np.asarray(row["roi_xyxy"], np.int64)
        full = np.asarray(row["full_image_size_wh"], np.int64)
        periodic = support["crop_policy_version"] == "mask_bbox_periodic_v2"
        center, size, _, rect = native_plan(box, roi, roi[2:] - roi[:2], full,
                                            periodic_x=periodic)
        if rect.tolist() != support["sampling_rect_erp"]:
            raise ValueError(f"support_changed:{row['observation_id']}")
        affine = gen_trans_from_patch_cv(center[0], center[1], size[0], size[1],
                                         384, 512, 1.0, 0)
        _, _, visibility, coordinate_valid = _targets(files, row["person_index"],
                                                       affine, roi, box, periodic)
        known = (visibility != -1) & coordinate_valid
        positive = int(np.count_nonzero(known & (visibility == 1)))
        negative = int(np.count_nonzero(known & (visibility == 0)))
        if index in selected and positive + negative != previous[order.index(index)]["counts"]["visibility"]["points"]:
            raise ValueError(f"historical_loss_mask_changed:{row['observation_id']}")
        kind = "both" if positive and negative else (
            "positive_only" if positive else "negative_only" if negative else "none")
        for population in ("all_train", "first_500") if index in selected else ("all_train",):
            for key in ("overall", category(row["episode"])):
                result = totals[(population, key)]
                result["samples"] += 1
                result["positive"] += positive
                result["negative"] += negative
                result[kind] += 1
    return {population: {key: totals[(population, key)] for key in
                         ("overall", "one_cycle", "two_cycle", "cycle_walk", "three_cycle")}
            for population in ("all_train", "first_500")}


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = summary()
    (OUTPUT / "supervision_counts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
