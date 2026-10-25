"""Add class-mean predictions for the frozen 319 real-riding targets, without changing old results."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mamma_ext.model import deployment_config, predict_one, restore_xy, strict_model_from_tensors
from mamma_ext.policy import load_active_policy
from mamma_ext.recovery import load_evaluation_boundary, load_frozen_20261024_class_mean_state
from scripts.real_riding_infer import EXPERIMENT, input_for, load_rows


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retrain/mamma_city_full_visibility_class_mean_20261024.json"
OUTPUT = ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024"
CHECKPOINT = OUTPUT / "train_resume_500/checkpoint_update_000500.pt"


def run(device, *, config_path=CONFIG, output=OUTPUT, checkpoint=CHECKPOINT,
        model_name="class_mean_500"):
    images, ready = load_rows()
    review = json.loads((EXPERIMENT / "review_data.json").read_text())
    ordered_ids = [row["target_id"] for row in review["targets"]]
    by_id = {row["target_id"]: row for row in ready}
    if len(ordered_ids) != 319 or set(ordered_ids) != set(by_id):
        raise ValueError("frozen_real_target_set_changed")
    rows = [by_id[key] for key in ordered_ids]
    with np.load(EXPERIMENT / "predictions_original.npz", allow_pickle=False) as original:
        old_ids = original["target_ids"].tolist()
        old_affine = original["input_affine_erp_to_model"].copy()
        vertex_ids = original["ordered_vertex_ids"].copy()
        template_id = original["template_id"].copy()
    with np.load(EXPERIMENT / "predictions_finetuned_500.npz", allow_pickle=False) as prior:
        if prior["target_ids"].tolist() != ordered_ids or not np.array_equal(
            prior["input_affine_erp_to_model"], old_affine
        ):
            raise ValueError("old_model_inputs_differ")
    with np.load(ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024/predictions_real_class_mean_500.npz",
                 allow_pickle=False) as archived:
        if archived["target_ids"].tolist() != ordered_ids or not np.array_equal(
            archived["input_affine_erp_to_model"], old_affine
        ):
            raise ValueError("archived_class_mean_inputs_differ")
        old_fingerprints = archived["input_fingerprints"].tolist()
    if old_ids != ordered_ids or not np.array_equal(vertex_ids, review["ordered_vertex_ids"]):
        raise ValueError("saved_real_prediction_template_or_order_changed")
    config, _ = load_active_policy(config_path, split="validation")
    from mamma_ext.policy import _config_path
    split_path = Path(config["input_config"])
    split = json.loads(split_path.read_text())
    manifests = [_config_path(split_path, p) for p in split["validation"]]
    model, report = strict_model_from_tensors(deployment_config(config["model_config"]),
                                               config["tensor_state"])
    if model is None or not report["strict_load"]:
        raise ValueError("original_model_strict_load_failed")
    if model_name == "class_mean_500":
        load_frozen_20261024_class_mean_state(checkpoint, model, config, manifests)
    elif model_name in ("class_mean_occlusion30_500", "class_mean_merged_occlusion30_500",
                        "class_mean_merged_full_epoch"):
        load_evaluation_boundary(checkpoint, model, config, manifests)
    else:
        raise ValueError("unsupported_fixed_real_comparison_model")
    model = model.to(device).eval().float()
    result_xy, raw_xy, logits, input_fingerprints, domains = [], [], [], [], []
    valid_masks = {}
    image_id, image = None, None
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if row["image_id"] != image_id:
                image_id = row["image_id"]
                image = cv2.imread(images[image_id]["source_png"], cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (2880, 5760):
                    raise ValueError(f"frozen_real_image_invalid:{image_id}")
            item, fingerprint = input_for(row, image)
            if fingerprint != old_fingerprints[index] or not np.array_equal(item["affine_erp_to_model"].astype(np.float32), old_affine[index]) or (
                item["parent_box_xyxy"].tolist() != review["targets"][index]["box"]
            ):
                raise ValueError(f"real_input_changed:{row['target_id']}")
            prediction, info = predict_one(model, item, device)
            if any(info["nonfinite_counts"].values()):
                raise ValueError(f"nonfinite_real_prediction:{row['target_id']}")
            raw = prediction["joints2d"][0, :, :2].astype(np.float32)
            _, erp, inverse = restore_xy(raw, item, plus_minus_one=True)
            delta = erp - inverse
            delta[:, 0] = (delta[:, 0] + 2880) % 5760 - 2880
            if np.max(np.abs(delta)) > .01:
                raise ValueError(f"real_coordinate_restore_mismatch:{row['target_id']}")
            logit = prediction["visibility"][0, :, 0].astype(np.float32)
            info_image = images[image_id]
            valid_key = (info_image["dataset"], info_image["camera"])
            if valid_key not in valid_masks:
                with np.load(info_image["valid_mask_resource"], allow_pickle=False) as source:
                    valid_masks[valid_key] = source[f"valid_masks_{info_image['camera']}"].copy()
            valid = valid_masks[valid_key]
            inside = np.isfinite(erp).all(1) & (erp[:, 0] >= 0) & (erp[:, 0] < 5760) & (
                erp[:, 1] >= 0) & (erp[:, 1] < 2880)
            domain = np.zeros(512, np.uint8)
            domain[inside] = 1
            indices = np.flatnonzero(inside)
            domain[indices] = np.where(valid[erp[indices, 1].astype(int),
                                             erp[indices, 0].astype(int)], 2, 1)
            result_xy.append(erp.astype(np.float32))
            raw_xy.append(raw)
            logits.append(logit)
            input_fingerprints.append(fingerprint)
            domains.append(domain)
            if (index + 1) % 40 == 0 or index + 1 == len(rows):
                print(f"{model_name}: {index + 1}/{len(rows)}", flush=True)
    result_xy = np.stack(result_xy)
    raw_xy = np.stack(raw_xy)
    logits = np.stack(logits)
    domains = np.stack(domains)
    probability = 1 / (1 + np.exp(-logits))
    output_npz = output / f"predictions_real_{model_name}.npz"
    output_json = output / f"review_predictions_{model_name}.json"
    if output_npz.exists() or output_json.exists():
        raise FileExistsError("balanced_real_predictions_already_exist")
    np.savez_compressed(output_npz, target_ids=np.asarray(ordered_ids), template_id=template_id,
                        ordered_vertex_ids=vertex_ids, raw_xy_model_normalized=raw_xy,
                        xy_full_erp=result_xy, visibility_logits=logits,
                        visibility_probability=probability, point_domain=domains,
                        input_affine_erp_to_model=old_affine,
                        input_fingerprints=np.asarray(input_fingerprints))
    display = {"schema": ("real_riding_class_mean_predictions_v1" if model_name == "class_mean_500"
                          else "real_riding_additional_predictions_v1"),
               "review_set_id": review["review_set_id"],
               "model": model_name,
               "checkpoint": str(checkpoint),
               "checkpoint_size_bytes": checkpoint.stat().st_size,
               "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
               "target_ids": ordered_ids,
               "targets": {key: {"xy": np.round(result_xy[i], 3).tolist(),
                                 "visibility_probability": np.round(probability[i], 5).tolist(),
                                 "point_domain": domains[i].tolist()}
                           for i, key in enumerate(ordered_ids)}}
    output_json.write_text(json.dumps(display, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps({"targets": len(rows), "visible_points_at_0.5": int((logits >= 0).sum()),
                      "gpu": str(device), "output": str(output_npz)}, ensure_ascii=False))


if __name__ == "__main__":
    run(torch.device("cuda:0"))
