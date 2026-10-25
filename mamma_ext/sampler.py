"""Freeze a short, reproducible occlusion-quota schedule from training GT only."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import _targets, gen_trans_from_patch_cv, native_plan


def label_counts(dataset, index):
    """Reproduce the reader's loss masks from saved GT, without opening RGB or mask PNGs."""
    slot = dataset.ready[index]
    row, files = dataset.slots[slot], dataset._files[slot]
    support = row["crop_support"]
    box = np.asarray(support["box_xyxy"], dtype=np.int64)
    roi = np.asarray(row["roi_xyxy"], dtype=np.int64)
    full = np.asarray(row["full_image_size_wh"], dtype=np.int64)
    periodic = support["crop_policy_version"] == "mask_bbox_periodic_v2"
    center, size, _, rect = native_plan(box, roi, roi[2:] - roi[:2], full, periodic_x=periodic)
    if rect.tolist() != support["sampling_rect_erp"]:
        raise ValueError("sampling_crop_support_changed:" + row["observation_id"])
    affine = gen_trans_from_patch_cv(center[0], center[1], size[0], size[1], 384, 512, 1.0, 0)
    _, _, visibility, coordinate_valid = _targets(
        files, row["person_index"], affine, roi, box, periodic
    )
    known = (visibility != -1) & coordinate_valid
    positive = int(np.count_nonzero(known & (visibility == 1)))
    negative = int(np.count_nonzero(known & (visibility == 0)))
    coordinates = int(np.count_nonzero(coordinate_valid))
    return {"positive": positive, "negative": negative, "coordinates": coordinates}


def _pool_round(dataset, indices, seed, pool_code, round_index):
    by_episode = defaultdict(list)
    for index in indices:
        slot = dataset.slots[dataset.ready[index]]
        by_episode[slot["episode"]].append(index)
    episodes = sorted(by_episode)
    streams = {}
    for episode_index, episode in enumerate(episodes):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, pool_code, round_index, episode_index]))
        streams[episode] = rng.permutation(sorted(by_episode[episode])).tolist()
    order = []
    turn = 0
    while any(streams.values()):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, pool_code, round_index, turn, 1]))
        for episode_index in rng.permutation(len(episodes)):
            episode = episodes[episode_index]
            if streams[episode]:
                order.append(streams[episode].pop())
        turn += 1
    return order


def _take_pool(dataset, indices, count, seed, pool_code):
    chosen = []
    round_index = 0
    while len(chosen) < count:
        remaining = count - len(chosen)
        chosen.extend(_pool_round(dataset, indices, seed, pool_code, round_index)[:remaining])
        round_index += 1
    return chosen


def _identity(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def mixed_order(dataset, seed, epoch):
    """Visit each episode once per round, shuffling within each episode."""
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("invalid_sampler_seed_or_epoch")
    groups = defaultdict(list)
    for index, slot in enumerate(dataset.ready):
        groups[dataset.slots[slot]["episode"]].append(index)
    episodes = sorted(groups)
    streams = {}
    for group_index, episode in enumerate(episodes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, group_index]))
        streams[episode] = rng.permutation(groups[episode]).tolist()
    order = []
    round_index = 0
    while any(streams.values()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, round_index, 1]))
        for group_index in rng.permutation(len(episodes)):
            episode = episodes[group_index]
            if streams[episode]:
                order.append(streams[episode].pop())
        round_index += 1
    if len(order) != len(dataset) or len(set(order)) != len(order):
        raise ValueError("invalid_mixed_sample_order")
    return order


def make_one_epoch_plan(dataset, seed):
    """Freeze one pass over all trainable observed targets, recording exclusions."""
    entries, excluded = [], []
    for index in mixed_order(dataset, seed, 0):
        row = dataset.slots[dataset.ready[index]]
        counts = label_counts(dataset, index)
        entry = {"dataset_index": index, "observation_id": row["observation_id"],
                 "episode": row["episode"], "person_id": row["person_id"],
                 "frame_index": row["frame_index"], **counts}
        (entries if counts["coordinates"] or counts["positive"] or counts["negative"]
         else excluded).append(entry)
    payload = {"schema": "mamma_one_epoch_plan_v1", "seed": seed,
               "train_manifests": dataset.cfg["manifests"],
               "entries": entries, "excluded_no_supervision": excluded}
    return {**payload, "plan_id": _identity(payload)}


def load_one_epoch_plan(path, dataset, seed):
    plan = json.loads(Path(path).read_text())
    plan_id = plan.pop("plan_id", None)
    if (plan.get("schema") != "mamma_one_epoch_plan_v1" or
        plan.get("seed") != seed or
        plan.get("train_manifests") != dataset.cfg["manifests"] or
        plan_id != _identity(plan)):
        raise ValueError("one_epoch_plan_identity_or_data_mismatch")
    actual = make_one_epoch_plan(dataset, seed)
    if plan != {key: value for key, value in actual.items() if key != "plan_id"}:
        raise ValueError("one_epoch_plan_gt_or_order_changed")
    return {**plan, "plan_id": plan_id}


def make_occlusion_plan(dataset, seed, total=500, occlusion_count=150):
    if (type(seed) is not int or seed < 0 or total != 500 or occlusion_count != 150):
        raise ValueError("unsupported_occlusion_quota")
    inventory = [label_counts(dataset, index) for index in range(len(dataset))]
    occlusion = [i for i, counts in enumerate(inventory) if counts["negative"] > 0]
    other = [i for i, counts in enumerate(inventory) if counts["negative"] == 0]
    trainable_other = [i for i in other if inventory[i]["coordinates"] or inventory[i]["positive"]]
    if not occlusion or not other or len(occlusion) + len(other) != len(dataset):
        raise ValueError("invalid_training_occlusion_pools")
    selected = {"occlusion": _take_pool(dataset, occlusion, occlusion_count, seed, 7),
                "other": _take_pool(dataset, trainable_other, total - occlusion_count, seed, 11)}
    schedule = []
    for block in range(total // 10):
        rng = np.random.default_rng(np.random.SeedSequence([seed, block, 23]))
        occlusion_positions = set(rng.choice(10, size=3, replace=False).tolist())
        schedule.extend("occlusion" if position in occlusion_positions else "other"
                        for position in range(10))
    cursors = Counter()
    entries = []
    for position, pool in enumerate(schedule):
        index = selected[pool][cursors[pool]]
        cursors[pool] += 1
        row = dataset.slots[dataset.ready[index]]
        entries.append({"position": position, "pool": pool, "dataset_index": index,
                        "observation_id": row["observation_id"], "episode": row["episode"],
                        "person_id": row["person_id"], "frame_index": row["frame_index"],
                        **inventory[index]})
    if cursors != Counter({"occlusion": 150, "other": 350}):
        raise AssertionError("sampling_quota_not_met")
    pool_inventory = {}
    for pool, indices in (("occlusion", occlusion), ("other", other)):
        by_episode = defaultdict(lambda: {"observations": 0, "positive": 0,
                                         "negative": 0, "coordinates": 0,
                                         "people": defaultdict(int), "frames": set()})
        for index in indices:
            row = dataset.slots[dataset.ready[index]]
            group = by_episode[row["episode"]]
            group["observations"] += 1
            for key in ("positive", "negative", "coordinates"):
                group[key] += inventory[index][key]
            group["people"][str(row["person_id"])] += 1
            group["frames"].add(row["frame_index"])
        pool_inventory[pool] = {
            episode: {**{key: value for key, value in group.items()
                         if key not in ("people", "frames")},
                      "people": dict(group["people"]),
                      "frames": sorted(group["frames"])}
            for episode, group in sorted(by_episode.items())}
    payload = {"schema": "mamma_occlusion_sampling_plan_v1", "seed": seed,
               "train_manifests": dataset.cfg["manifests"],
               "pool_sizes": {"occlusion": len(occlusion), "other": len(other)},
               "untrainable_other": len(other) - len(trainable_other),
               "pool_inventory": pool_inventory,
               "entries": entries}
    return {**payload, "plan_id": _identity(payload)}


def load_occlusion_plan(path, dataset, seed, total=500, occlusion_count=150):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    identity = plan.pop("plan_id", None)
    if (plan.get("schema") != "mamma_occlusion_sampling_plan_v1" or
        plan.get("seed") != seed or plan.get("train_manifests") != dataset.cfg["manifests"] or
        identity != _identity(plan) or len(plan.get("entries", [])) != total):
        raise ValueError("sampling_plan_identity_or_data_mismatch")
    entries = plan["entries"]
    if (sum(row["pool"] == "occlusion" for row in entries) != occlusion_count or
        sum(row["pool"] == "other" for row in entries) != total - occlusion_count):
        raise ValueError("sampling_plan_quota_mismatch")
    for position, entry in enumerate(entries):
        index = entry["dataset_index"]
        if type(index) is not int or not 0 <= index < len(dataset):
            raise ValueError("sampling_plan_index_invalid")
        row = dataset.slots[dataset.ready[index]]
        if (entry["position"] != position or entry["observation_id"] != row["observation_id"] or
            entry["episode"] != row["episode"] or entry["person_id"] != row["person_id"] or
            entry["frame_index"] != row["frame_index"] or
            entry["pool"] != ("occlusion" if entry["negative"] else "other") or
            entry["positive"] + entry["negative"] + entry["coordinates"] == 0):
            raise ValueError("sampling_plan_entry_mismatch")
    checked = {}
    for entry in entries:
        index = entry["dataset_index"]
        if index not in checked:
            checked[index] = label_counts(dataset, index)
        if any(entry[key] != checked[index][key] for key in checked[index]):
            raise ValueError("sampling_plan_gt_changed")
    return {**plan, "plan_id": identity}
