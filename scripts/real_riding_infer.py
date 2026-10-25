"""Run the two fixed MAMMA states on frozen real-riding people without GT."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from mamma_ext.data import real_erp_input
from mamma_ext.model import deployment_config, predict_one, restore_xy, strict_model_from_tensors
from mamma_ext.recovery import load_frozen_20261024_point_mean_state


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
CONFIG = ROOT / "configs/retrain/mamma_city_full_20261024.json"
CHECKPOINT = ROOT / "outputs/experiments/city_full_first_finetune_20261024/train_resume_500/checkpoint_update_000500.pt"
ASSIGNMENT_SEED = 20261024


def training_contract():
    path = CONFIG.resolve()
    config = json.loads(path.read_text())
    for name in ("input_config", "model_config", "tensor_state"):
        config[name] = str((path.parent / config[name]).resolve())
    split_file = Path(config["input_config"])
    split = json.loads(split_file.read_text())
    validation = [str((split_file.parent / p).resolve()) for p in split["validation"]]
    return config, validation


def load_rows():
    frozen = json.loads((EXPERIMENT / "targets_ready.json").read_text())
    selection = json.loads((EXPERIMENT / "selection.json").read_text())
    if frozen["schema"] != "real_riding_ready_targets_v1" or selection["schema"] != "real_riding_selection_v1":
        raise ValueError("input_selection_not_frozen")
    images = {row["image_id"]: row for row in selection["images"]}
    if len(images) != 80 or len(frozen["targets"]) != len({r["target_id"] for r in frozen["targets"]}):
        raise ValueError("invalid_frozen_target_set")
    return images, frozen["targets"]


def input_for(row, image):
    mask = cv2.imread(str(EXPERIMENT / row["mask"]), cv2.IMREAD_GRAYSCALE)
    item = real_erp_input(image, mask)
    digest = hashlib.blake2b(digest_size=20)
    for key in ("img", "mask", "affine_erp_to_model", "parent_box_xyxy"):
        array = np.ascontiguousarray(item[key])
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return item, digest.hexdigest()


def infer_state(model, device, images, rows, expected_inputs=None):
    points, logits, fingerprints, boxes = [], [], [], []
    model.eval()
    current_image_id, image = None, None
    for index, row in enumerate(rows, 1):
        if row["image_id"] != current_image_id:
            current_image_id = row["image_id"]
            image = cv2.imread(images[current_image_id]["source_png"], cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (2880, 5760):
                raise ValueError(f"source_image_missing_or_wrong_size:{current_image_id}")
        item, digest = input_for(row, image)
        if expected_inputs is not None and digest != expected_inputs[index - 1]:
            raise ValueError(f"model_input_changed:{row['target_id']}")
        prediction, info = predict_one(model, item, device)
        if any(count for count in info["nonfinite_counts"].values()):
            raise ValueError(f"nonfinite_model_output:{row['target_id']}")
        raw_xy = prediction["joints2d"][0, :, :2]
        _, deployment_xy, inverse_xy = restore_xy(raw_xy, item, plus_minus_one=True)
        if deployment_xy.shape != (512, 2) or np.max(np.abs((deployment_xy[:, 0] - inverse_xy[:, 0] + 2880) % 5760 - 2880)) > .01 or np.max(np.abs(deployment_xy[:, 1] - inverse_xy[:, 1])) > .01:
            raise ValueError(f"restore_methods_disagree:{row['target_id']}")
        point_logits = prediction["visibility"][0, :, 0]
        if point_logits.shape != (512,):
            raise ValueError(f"visibility_shape_conflict:{row['target_id']}")
        points.append(deployment_xy.astype(np.float32))
        logits.append(point_logits.astype(np.float32))
        fingerprints.append(digest)
        boxes.append(item["parent_box_xyxy"].tolist())
        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} targets", flush=True)
    return np.stack(points), np.stack(logits), fingerprints, boxes


def write_outputs(images, rows, original, fine, fingerprints, boxes, gpu):
    orig_xy, orig_logits = original
    fine_xy, fine_logits = fine
    target_ids = np.asarray([r["target_id"] for r in rows])
    for name, xy, logits in (("original", orig_xy, orig_logits), ("finetuned_500", fine_xy, fine_logits)):
        np.savez_compressed(EXPERIMENT / f"predictions_{name}.npz", target_ids=target_ids,
                            xy_full_erp=xy, visibility_logits=logits,
                            visibility_probability=1 / (1 + np.exp(-logits)))
    rng = random.Random(ASSIGNMENT_SEED)
    assignment = {row["target_id"]: ("original" if rng.getrandbits(1) else "finetuned_500") for row in rows}
    original_file = ROOT / "outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt"
    model_files = {"original": original_file, "finetuned_500": CHECKPOINT}
    model_identity = {name: {"path": str(file), "size_bytes": file.stat().st_size,
                             "mtime_ns": file.stat().st_mtime_ns} for name, file in model_files.items()}
    mapping = {"schema": "real_riding_blind_mapping_v1", "review_set_id": "real_riding_20261024_fixed80_v1",
               "seed": ASSIGNMENT_SEED, "A_by_target": assignment,
               "model_identity": model_identity}
    (EXPERIMENT / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    entries = []
    shifts = []
    flips = []
    for index, row in enumerate(rows):
        a_orig = assignment[row["target_id"]] == "original"
        a_xy, b_xy = (orig_xy[index], fine_xy[index]) if a_orig else (fine_xy[index], orig_xy[index])
        a_logit, b_logit = (orig_logits[index], fine_logits[index]) if a_orig else (fine_logits[index], orig_logits[index])
        dx = (orig_xy[index, :, 0] - fine_xy[index, :, 0] + 2880) % 5760 - 2880
        dy = orig_xy[index, :, 1] - fine_xy[index, :, 1]
        shifts.append(float(np.median(np.hypot(dx, dy))))
        flips.append(int(((orig_logits[index] >= 0) != (fine_logits[index] >= 0)).sum()))
        entries.append({"target_id": row["target_id"], "image_id": row["image_id"],
                        "raw_index": row["raw_index"], "mask": row["mask"], "box": boxes[index],
                        "A": {"xy": np.round(a_xy, 3).tolist(), "visibility_probability": np.round(1 / (1 + np.exp(-a_logit)), 5).tolist()},
                        "B": {"xy": np.round(b_xy, 3).tolist(), "visibility_probability": np.round(1 / (1 + np.exp(-b_logit)), 5).tolist()}})
    review = {"schema": "real_riding_review_data_v1", "review_set_id": mapping["review_set_id"],
              "assignment_seed": ASSIGNMENT_SEED, "model_identity": model_identity,
              "visibility_threshold": .5,
              "notice": "无人工 512 点真值；A/B 之间的差异不是精度。遮挡点位置不一定能从 RGB 判断。",
              "images": list(images.values()), "targets": entries,
              "input_failures": json.loads((EXPERIMENT / "targets_ready.json").read_text())["rejected"]}
    (EXPERIMENT / "review_data.json").write_text(json.dumps(review, ensure_ascii=False, separators=(",", ":")))
    diagnostics = {"selected_images": len(images), "target_count": len(rows), "gpu": gpu,
                   "identical_input_fingerprints": len(fingerprints),
                   "median_of_target_median_model_displacement_px": float(np.median(shifts)),
                   "visibility_threshold_flips": int(sum(flips)),
                   "meaning": "model difference only, not GT error or recall"}
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")


def annotate_prediction_domain(review):
    """Mark original-image and calibrated physical-domain support, never clamp XY."""
    image_rows = {row["image_id"]: row for row in review["images"]}
    valid = {}
    for image in image_rows.values():
        key = (image["dataset"], image["camera"])
        if key not in valid:
            with np.load(image["valid_mask_resource"], allow_pickle=False) as data:
                valid[key] = data[f"valid_masks_{image['camera']}"].copy()
    for entry in review["targets"]:
        image = image_rows[entry["image_id"]]
        mask = valid[(image["dataset"], image["camera"])]
        for side in ("A", "B"):
            points = np.asarray(entry[side]["xy"], dtype=np.float64)
            inside = np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] < 5760) & (points[:, 1] >= 0) & (points[:, 1] < 2880)
            domain = np.zeros(512, np.uint8)
            domain[inside] = 1
            indices = np.flatnonzero(inside)
            domain[indices] = np.where(mask[points[indices, 1].astype(int), points[indices, 0].astype(int)], 2, 1)
            entry[side]["point_domain"] = domain.tolist()


def supplement_vertical_targets(device):
    """Infer only targets newly admitted by the explicit real-ERP black-pad policy."""
    images, ready = load_rows()
    review_file = EXPERIMENT / "review_data.json"
    review = json.loads(review_file.read_text())
    mapping_file = EXPERIMENT / "mapping.json"
    mapping = json.loads(mapping_file.read_text())
    old_ids = [row["target_id"] for row in review["targets"]]
    ready_ids = {row["target_id"] for row in ready}
    if len(old_ids) != 314 or len(ready) != 319 or not set(old_ids).issubset(ready_ids):
        raise ValueError("unexpected_original_or_supplement_target_set")
    extra = [row for row in ready if row["target_id"] not in set(old_ids)]
    if len(extra) != 5 or any(not any(row["vertical_padding_top_bottom"]) for row in extra):
        raise ValueError("supplement_requires_five_vertical_padding_targets")
    cfg, validation = training_contract()
    model_cfg = deployment_config(cfg["model_config"])
    model, report = strict_model_from_tensors(model_cfg, cfg["tensor_state"])
    if model is None or not report["strict_load"]:
        raise ValueError("original_weight_mismatch")
    model = model.to(device).eval().float()
    orig_xy, orig_logits, fingerprints, boxes = infer_state(model, device, images, extra)
    load_frozen_20261024_point_mean_state(CHECKPOINT, model, cfg, validation)
    model.eval().float()
    fine_xy, fine_logits, _, fine_boxes = infer_state(model, device, images, extra, fingerprints)
    if boxes != fine_boxes:
        raise ValueError("supplement_input_boxes_changed")
    original_file = EXPERIMENT / "predictions_original.npz"
    fine_file = EXPERIMENT / "predictions_finetuned_500.npz"
    with np.load(original_file, allow_pickle=False) as saved:
        old_original = {name: saved[name].copy() for name in saved.files}
    with np.load(fine_file, allow_pickle=False) as saved:
        old_fine = {name: saved[name].copy() for name in saved.files}
    if old_original["target_ids"].tolist() != old_ids or old_fine["target_ids"].tolist() != old_ids:
        raise ValueError("saved_prediction_order_conflict")
    if set(mapping["A_by_target"]) != set(old_ids):
        raise ValueError("saved_blind_mapping_conflict")
    for name, path, old, new_xy, new_logits in (
        ("original", original_file, old_original, orig_xy, orig_logits),
        ("finetuned_500", fine_file, old_fine, fine_xy, fine_logits),
    ):
        temporary = path.with_name(path.stem + ".partial.npz")
        np.savez_compressed(
            temporary,
            target_ids=np.r_[old["target_ids"], [row["target_id"] for row in extra]],
            xy_full_erp=np.concatenate((old["xy_full_erp"], new_xy)),
            visibility_logits=np.concatenate((old["visibility_logits"], new_logits)),
            visibility_probability=np.concatenate((old["visibility_probability"], 1 / (1 + np.exp(-new_logits)))),
        )
    by_id = {row["target_id"]: row for row in ready}
    for entry in review["targets"]:
        entry["vertical_padding_top_bottom"] = by_id[entry["target_id"]]["vertical_padding_top_bottom"]
    for index, row in enumerate(extra):
        target_id = row["target_id"]
        a_original = bool(random.Random(f"{ASSIGNMENT_SEED}:{target_id}").getrandbits(1))
        mapping["A_by_target"][target_id] = "original" if a_original else "finetuned_500"
        a_xy, b_xy = (orig_xy[index], fine_xy[index]) if a_original else (fine_xy[index], orig_xy[index])
        a_logit, b_logit = (orig_logits[index], fine_logits[index]) if a_original else (fine_logits[index], orig_logits[index])
        review["targets"].append({
            "target_id": target_id, "image_id": row["image_id"], "raw_index": row["raw_index"],
            "mask": row["mask"], "box": boxes[index],
            "vertical_padding_top_bottom": row["vertical_padding_top_bottom"],
            "A": {"xy": np.round(a_xy, 3).tolist(), "visibility_probability": np.round(1 / (1 + np.exp(-a_logit)), 5).tolist()},
            "B": {"xy": np.round(b_xy, 3).tolist(), "visibility_probability": np.round(1 / (1 + np.exp(-b_logit)), 5).tolist()},
        })
    review["input_failures"] = json.loads((EXPERIMENT / "targets_ready.json").read_text())["rejected"]
    annotate_prediction_domain(review)
    for path, data in ((review_file, review), (mapping_file, mapping)):
        path.with_name(path.stem + ".partial.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    for path in (original_file, fine_file):
        os.replace(path.with_name(path.stem + ".partial.npz"), path)
    for path in (review_file, mapping_file):
        os.replace(path.with_name(path.stem + ".partial.json"), path)
    diagnostics_file = EXPERIMENT / "diagnostics.json"
    diagnostics = json.loads(diagnostics_file.read_text())
    diagnostics.update(target_count=len(review["targets"]), identical_input_fingerprints=len(review["targets"]),
                       supplemented_vertical_padding=[{"target_id": row["target_id"], "top_bottom_px": row["vertical_padding_top_bottom"]} for row in extra],
                       input_failures=review["input_failures"])
    diagnostics_file.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    print(f"Added {len(extra)} vertical-border targets; original {len(old_ids)} predictions reused")


def capture_raw_coordinates(device):
    """Archive the exact normalized model XY alongside already-verified ERP XY."""
    images, ready = load_rows()
    review = json.loads((EXPERIMENT / "review_data.json").read_text())
    ordered_ids = [row["target_id"] for row in review["targets"]]
    by_id = {row["target_id"]: row for row in ready}
    if len(ordered_ids) != 319 or set(ordered_ids) != set(by_id):
        raise ValueError("raw_capture_target_set_conflict")
    rows = [by_id[target_id] for target_id in ordered_ids]
    cfg, validation = training_contract()
    model_cfg = deployment_config(cfg["model_config"])
    model, report = strict_model_from_tensors(model_cfg, cfg["tensor_state"])
    if model is None or not report["strict_load"]:
        raise ValueError("original_weight_mismatch")
    model = model.to(device).eval().float()
    previous_fingerprints = None
    completed = []
    for state_name in ("original", "finetuned_500"):
        if state_name == "finetuned_500":
            load_frozen_20261024_point_mean_state(CHECKPOINT, model, cfg, validation)
            model.eval().float()
        path = EXPERIMENT / f"predictions_{state_name}.npz"
        with np.load(path, allow_pickle=False) as data:
            saved = {name: data[name].copy() for name in data.files}
        if saved["target_ids"].tolist() != ordered_ids:
            raise ValueError("raw_capture_saved_order_conflict")
        raw_values, affines, fingerprints = [], [], []
        current_image_id, image = None, None
        for index, row in enumerate(rows):
            if row["image_id"] != current_image_id:
                current_image_id = row["image_id"]
                image = cv2.imread(images[current_image_id]["source_png"], cv2.IMREAD_COLOR)
            item, digest = input_for(row, image)
            if previous_fingerprints is not None and digest != previous_fingerprints[index]:
                raise ValueError(f"raw_capture_input_changed:{row['target_id']}")
            prediction, info = predict_one(model, item, device)
            if any(info["nonfinite_counts"].values()):
                raise ValueError(f"raw_capture_nonfinite:{row['target_id']}")
            raw_xy = prediction["joints2d"][0, :, :2].astype(np.float32)
            _, restored, _ = restore_xy(raw_xy, item, plus_minus_one=True)
            if not np.allclose(restored.astype(np.float32), saved["xy_full_erp"][index], rtol=0, atol=.001):
                raise ValueError(f"raw_capture_erp_changed:{row['target_id']}")
            if not np.allclose(prediction["visibility"][0, :, 0], saved["visibility_logits"][index], rtol=0, atol=.001):
                raise ValueError(f"raw_capture_visibility_changed:{row['target_id']}")
            raw_values.append(raw_xy)
            affines.append(item["affine_erp_to_model"].astype(np.float32))
            fingerprints.append(digest)
            if (index + 1) % 40 == 0 or index + 1 == len(rows):
                print(f"{state_name} raw: {index + 1}/{len(rows)}", flush=True)
        saved["raw_xy_model_normalized"] = np.stack(raw_values)
        saved["input_affine_erp_to_model"] = np.stack(affines)
        completed.append((path, saved))
        previous_fingerprints = fingerprints
    for path, saved in completed:
        temporary = path.with_name(path.stem + ".raw.partial.npz")
        np.savez_compressed(temporary, **saved)
        os.replace(temporary, path)
    print("Archived exact raw normalized XY; existing ERP and visibility arrays rechecked")


def point_template_order():
    """Read only the fixed point order already used by the 500-step training."""
    config, _ = training_contract()
    split_file = Path(config["input_config"])
    split = json.loads(split_file.read_text())
    template_id, ordered = None, None
    for reference in split["train"] + split["validation"]:
        manifest = json.loads((split_file.parent / reference).resolve().read_text())
        current = np.asarray(manifest["ordered_vertex_ids"], dtype=np.int32)
        if manifest["template_id"] != "MAMMA_verts_512_ordered_v1" or current.shape != (512,):
            raise ValueError("real_review_training_template_invalid")
        if ordered is not None and not np.array_equal(ordered, current):
            raise ValueError("real_review_training_template_changed")
        template_id, ordered = manifest["template_id"], current
    return template_id, ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--supplement-vertical", action="store_true")
    parser.add_argument("--finalize-archive", action="store_true")
    parser.add_argument("--capture-raw", action="store_true")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.finalize_archive:
        path = EXPERIMENT / "review_data.json"
        review = json.loads(path.read_text())
        annotate_prediction_domain(review)
        template_id, vertex_ids = point_template_order()
        review["template_id"] = template_id
        review["ordered_vertex_ids"] = vertex_ids.tolist()
        path.write_text(json.dumps(review, ensure_ascii=False, separators=(",", ":")))
        with np.load(EXPERIMENT / "predictions_original.npz", allow_pickle=False) as saved:
            original_xy = saved["xy_full_erp"].copy()
            original_logits = saved["visibility_logits"].copy()
        with np.load(EXPERIMENT / "predictions_finetuned_500.npz", allow_pickle=False) as saved:
            final_xy = saved["xy_full_erp"].copy()
            final_logits = saved["visibility_logits"].copy()
        dx = (original_xy[:, :, 0] - final_xy[:, :, 0] + 2880) % 5760 - 2880
        dy = original_xy[:, :, 1] - final_xy[:, :, 1]
        diagnostics_file = EXPERIMENT / "diagnostics.json"
        diagnostics = json.loads(diagnostics_file.read_text())
        diagnostics.update(
            target_count=len(review["targets"]),
            median_of_target_median_model_displacement_px=float(np.median(np.median(np.hypot(dx, dy), axis=1))),
            visibility_threshold_flips=int(((original_logits >= 0) != (final_logits >= 0)).sum()),
            gpu_uuid=args.gpu_uuid,
            note="Both sequential model passes used one explicitly selected physical GPU",
        )
        diagnostics_file.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
        for name in ("original", "finetuned_500"):
            prediction_path = EXPERIMENT / f"predictions_{name}.npz"
            with np.load(prediction_path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays["template_id"] = np.asarray(template_id)
            arrays["ordered_vertex_ids"] = vertex_ids
            temporary = prediction_path.with_name(prediction_path.stem + ".final.partial.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, prediction_path)
        print(f"Marked image/physical domain for {len(review['targets'])} targets")
        return
    if args.capture_raw:
        capture_raw_coordinates(torch.device(args.device))
        return
    if args.supplement_vertical:
        supplement_vertical_targets(torch.device(args.device))
        return
    if any((EXPERIMENT / name).exists() for name in ("mapping.json", "review_data.json", "predictions_original.npz", "predictions_finetuned_500.npz")):
        raise FileExistsError("review_predictions_already_exist")
    images, rows = load_rows()
    if not rows:
        raise ValueError("no_frozen_targets")
    cfg, validation = training_contract()
    model_cfg = deployment_config(cfg["model_config"])
    model, report = strict_model_from_tensors(model_cfg, cfg["tensor_state"])
    if model is None or not report["strict_load"]:
        raise ValueError("original_weight_mismatch")
    device = torch.device(args.device)
    model = model.to(device).eval().float()
    print(f"original strict load {report['model_key_count']} keys", flush=True)
    original_xy, original_logits, fingerprints, boxes = infer_state(model, device, images, rows)
    load_frozen_20261024_point_mean_state(CHECKPOINT, model, cfg, validation)
    model.eval().float()
    print("finetuned checkpoint strict load passed", flush=True)
    fine_xy, fine_logits, _, fine_boxes = infer_state(model, device, images, rows, fingerprints)
    if boxes != fine_boxes:
        raise ValueError("model_input_boxes_changed")
    write_outputs(images, rows, (original_xy, original_logits), (fine_xy, fine_logits), fingerprints, boxes, str(device))


if __name__ == "__main__":
    main()
