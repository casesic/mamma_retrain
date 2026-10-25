"""Replay frozen real-riding predictions through the existing D/E demo stitcher.

This reads the two saved PyTorch prediction files and the original ERP PNGs.
No detector or network is loaded.  The read-only riding-data and mamma-deploy
modules supply the same D pairing, directional gate, local warp and hard cover
used by the delivered demonstration videos.  That warp is not canonical CPW.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
FULL_EPOCH = ROOT / "outputs/experiments/city_merged_one_epoch_20261025"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/real_riding_stitch_threshold085_20261025"
PREVIEW_SIZE = (1440, 720)
MODEL_WEIGHTS = {
    "original": ROOT / "outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt",
    "full_epoch": FULL_EPOCH / "train_full/checkpoint_update_000812.pt",
}
CAMERAS = ("cam2", "cam4")
BRANCHES = (("original_050", "original", 0.5),
            ("full_epoch_085", "full_epoch", 0.85),
            ("full_epoch_050", "full_epoch", 0.5))


def read_predictions():
    paths = {
        "original": REVIEW / "predictions_original.npz",
        "full_epoch": FULL_EPOCH / "predictions_real_class_mean_merged_full_epoch.npz",
    }
    predictions = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as file:
            predictions[name] = {key: file[key].copy() for key in file.files}
    original, trained = predictions.values()
    for key in ("target_ids", "ordered_vertex_ids", "input_affine_erp_to_model"):
        if not np.array_equal(original[key], trained[key]):
            raise ValueError(f"prediction_input_mismatch:{key}")
    if str(original["template_id"]) != str(trained["template_id"]):
        raise ValueError("prediction_template_mismatch")
    for name, data in predictions.items():
        xy = data["xy_full_erp"]
        probability = data["visibility_probability"]
        if xy.shape != (319, 512, 2) or probability.shape != (319, 512):
            raise ValueError(f"prediction_shape:{name}")
        if not np.isfinite(xy).all() or not np.isfinite(probability).all():
            raise ValueError(f"prediction_nonfinite:{name}")
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"prediction_probability_range:{name}")
    return predictions, paths


def frozen_inputs():
    selection = json.loads((REVIEW / "selection.json").read_text())
    target_file = json.loads((REVIEW / "targets.json").read_text())
    predictions, paths = read_predictions()
    images = selection["images"]
    targets = target_file["targets"]
    ids = [row["target_id"] for row in targets]
    if len(images) != 80 or len(targets) != 319 or len(set(ids)) != 319:
        raise ValueError("frozen_input_count_changed")
    prediction_ids = predictions["original"]["target_ids"].tolist()
    if set(ids) != set(prediction_ids):
        raise ValueError("frozen_target_identity_mismatch")
    prediction_index = {target_id: index for index, target_id in enumerate(prediction_ids)}
    by_image = defaultdict(list)
    for row in targets:
        by_image[row["image_id"]].append((prediction_index[row["target_id"]], row))
    pairs = defaultdict(dict)
    for image in images:
        pair_key = (image["dataset"], image["source_frame"])
        camera = image["camera"]
        if camera not in CAMERAS or camera in pairs[pair_key]:
            raise ValueError("frozen_camera_duplicate")
        pairs[pair_key][camera] = image
    if len(pairs) != 40 or any(set(pair) != set(CAMERAS) for pair in pairs.values()):
        raise ValueError("frozen_pair_coverage_mismatch")
    return sorted(pairs.items()), by_image, predictions, paths


def external_modules(riding_data, deployment_source, densekp_vendor):
    for path in (riding_data, deployment_source, densekp_vendor):
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))
    from mamma_point_pair.pairing import pair_current
    from mamma_pair_refinement import gate_warp
    from video_comparison_render_native import NativeComposite, render_mamma
    from video_baseline import load_masks
    return pair_current, gate_warp, NativeComposite, render_mamma, load_masks


def load_camera_data(image, indexed_targets):
    frame = cv2.imread(image["source_png"], cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (2880, 5760, 3):
        raise ValueError(f"invalid_erp_image:{image['image_id']}")
    people = []
    for index, target in indexed_targets:
        mask_path = (REVIEW / target["mask"]).resolve()
        if not mask_path.is_relative_to(REVIEW.resolve()):
            raise ValueError("mask_path_outside_frozen_review")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != frame.shape[:2] or not np.any(mask):
            raise ValueError(f"invalid_target_mask:{target['target_id']}")
        people.append((index, target, mask))
    return frame, people


def observations(people, prediction, threshold):
    result = []
    for index, target, mask in people:
        xy = prediction["xy_full_erp"][index]
        probability = prediction["visibility_probability"][index]
        points = np.column_stack((xy, np.zeros(512, np.float32)))
        result.append({
            "raw_index": int(target["raw_index"]), "source": "yolo_current",
            "role": "target", "mask": mask, "target_id": target["target_id"],
            "mamma": {"points": points,
                      "vis": (probability >= threshold).astype(np.float32),
                      "box": target["box_xyxy"], "score": float(target["score"])},
        })
    return result


def render_branch(camera_data, prediction, threshold, full_valid, small_valid,
                  composite, pairing, gate, render):
    erp2, people2 = camera_data["cam2"]
    erp4, people4 = camera_data["cam4"]
    observations_by_camera = [observations(people2, prediction, threshold),
                              observations(people4, prediction, threshold)]
    decision = pairing(observations_by_camera, full_valid)
    groups, masks, admission = [], [], []
    for edge in decision["pairs"]:
        i, j = edge["i"], edge["j"]
        p2 = decision["arrays"][0]["landmarks"][i]
        p4 = decision["arrays"][1]["landmarks"][j]
        keep = decision["keep"][i, j]
        group = np.column_stack((p4[keep, :2], p2[keep, :2])).astype(np.float32)
        reason, details = gate(group, "cam4_to_cam2", small_valid)
        admission.append({"cam2_target": decision["observations"][0][i]["target_id"],
                          "cam4_target": decision["observations"][1][j]["target_id"],
                          "common_points": int(keep.sum()), "gate": reason,
                          "details": details})
        if reason == "admit":
            groups.append(group)
            masks.append({"cam2": decision["observations"][0][i]["mask"],
                          "cam4": decision["observations"][1][j]["mask"]})
    if groups:
        image, fitted, failures, ownership = render(
            erp2, erp4, full_valid, groups, masks, small_valid, composite)
    else:
        image = composite.combine(erp2, erp4, None)
        fitted, failures, ownership = [], [], []
    if image.shape != erp2.shape:
        raise ValueError("stitch_image_shape_changed")
    reasons = Counter(reason for edge in decision["candidates"]
                      for reason in edge["reject_reasons"])
    record = {
        "observations": {camera: len(rows) for camera, rows in
                         zip(CAMERAS, observations_by_camera)},
        "candidate_rejections": dict(reasons),
        "person_pairs": [{"cam2_target": decision["observations"][0][e["i"]]["target_id"],
                          "cam4_target": decision["observations"][1][e["j"]]["target_id"],
                          "common_points": e["common_points"]} for e in decision["pairs"]],
        "warp_admission": admission, "warped": fitted, "warp_failures": failures,
        "ownership_fallback": ownership,
        "status": ("WARPED" if fitted and not failures else
                   "PARTIAL WARP" if fitted else
                   "WARP FAILED" if failures else
                   "WARP REJECTED" if decision["pairs"] else
                   "NO DETECTION" if not all(observations_by_camera) else "NO MATCH"),
    }
    return image, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-pairs", type=int, default=40)
    parser.add_argument("--riding-data-root", type=Path,
                        default=Path("/home/yy/Repos/riding-data"))
    parser.add_argument("--deployment-source", type=Path,
                        default=Path("/home/yy/Repos/mamma-deploy/src"))
    parser.add_argument("--densekp-vendor", type=Path,
                        default=Path("/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/dist/human_fitting_delivery/vendor"))
    args = parser.parse_args()
    if not 1 <= args.limit_pairs <= 40:
        parser.error("--limit-pairs must be between 1 and 40")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    pair_frames, by_image, predictions, prediction_paths = frozen_inputs()
    if any(not path.is_file() for path in MODEL_WEIGHTS.values()):
        raise FileNotFoundError("model_weight_provenance_missing")
    pair_current, gate_warp, NativeComposite, render_mamma, load_masks = external_modules(
        args.riding_data_root, args.deployment_source, args.densekp_vendor)
    output.mkdir(parents=True)
    cv2.setNumThreads(4)
    rows = []
    masks_by_calibration = {}
    started = time.monotonic()
    for pair_index, (pair_key, source) in enumerate(pair_frames[:args.limit_pairs]):
        dataset, source_frame = pair_key
        resources = {source[cam]["valid_mask_resource"] for cam in CAMERAS}
        calibrations = {source[cam]["calibration"] for cam in CAMERAS}
        if len(resources) != 1 or len(calibrations) != 1:
            raise ValueError(f"pair_calibration_mismatch:{pair_key}")
        calibration = calibrations.pop()
        if (dataset, calibration) not in (("data3", "20260320"), ("data4", "20260311")):
            raise ValueError(f"wrong_dataset_calibration:{pair_key}:{calibration}")
        if calibration not in masks_by_calibration:
            full_valid, small_valid = load_masks(Path(resources.pop()), (1920, 960))
            masks_by_calibration[calibration] = (
                full_valid, small_valid, NativeComposite(full_valid))
        full_valid, small_valid, composite = masks_by_calibration[calibration]
        camera_data = {cam: load_camera_data(source[cam], by_image[source[cam]["image_id"]])
                       for cam in CAMERAS}
        destination = output / "frames" / f"{dataset}_{source_frame}"
        preview_destination = output / "previews" / f"{dataset}_{source_frame}"
        destination.mkdir(parents=True)
        preview_destination.mkdir(parents=True)
        branch_rows = {}
        for branch, model, threshold in BRANCHES:
            image, detail = render_branch(camera_data, predictions[model], threshold,
                                          full_valid, small_valid, composite,
                                          pair_current, gate_warp, render_mamma)
            filename = destination / f"{branch}.png"
            if not cv2.imwrite(str(filename), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"cannot_write_image:{filename}")
            preview_filename = preview_destination / f"{branch}.png"
            preview = cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(preview_filename), preview,
                               [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"cannot_write_preview:{preview_filename}")
            detail.update({"model": model, "visibility_threshold": threshold,
                           "image": str(filename.relative_to(output)),
                           "preview": str(preview_filename.relative_to(output))})
            branch_rows[branch] = detail
        row = {"dataset": dataset, "source_frame": source_frame,
               "calibration": calibration,
               "cam2_image_id": source["cam2"]["image_id"],
               "cam4_image_id": source["cam4"]["image_id"],
               "branches": branch_rows}
        rows.append(row)
        print(f"{pair_index + 1}/{args.limit_pairs} {dataset}/{source_frame} " +
              " ".join(f"{branch}:{branch_rows[branch]['status']}"
                       for branch, _, _ in BRANCHES), flush=True)
    result = {
        "schema": "real_riding_demo_stitch_comparison_v1",
        "notice": "Read-only replay of the simplified riding-data demo stitcher; not canonical CPW or the company stitcher.",
        "input_selection": str((REVIEW / "selection.json").resolve()),
        "input_targets": str((REVIEW / "targets.json").resolve()),
        "prediction_files": {name: str(path.resolve()) for name, path in prediction_paths.items()},
        "model_weights": {name: str(path.resolve()) for name, path in MODEL_WEIGHTS.items()},
        "external_implementation": {
            "pairing": str((args.deployment_source / "mamma_point_pair/pairing.py").resolve()),
            "direction_gate": str((args.riding_data_root / "mamma_pair_refinement.py").resolve()),
            "mesh_ownership_cover": str((args.riding_data_root / "video_comparison_render_native.py").resolve()),
            "direction": "cam4_to_cam2", "output_size": [5760, 2880]},
        "branches": {name: {"model": model, "threshold": threshold}
                     for name, model, threshold in BRANCHES},
        "pairs": rows, "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (output / "index.json").write_text(json.dumps(result, ensure_ascii=False,
                                                   indent=2, allow_nan=False) + "\n")
    print(f"index: {output / 'index.json'}", flush=True)


if __name__ == "__main__":
    main()
