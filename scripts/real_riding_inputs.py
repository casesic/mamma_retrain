"""Freeze riding ERP inputs and segment people before comparing MAMMA weights.

The source images and the YOLO engine are read only. This script never loads MAMMA.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
RIDING = Path("/home/yy/Repos/riding-data")
ENGINE = Path("/home/yy/Repos/yolo-eval/yolo26s-seg_fp32.engine")
VALID = {
    "data3": Path("/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/tmp/production_regression_20260320_dataset3_multiperson/prepared/valid_masks_20260320.npz"),
    "data4": RIDING / "data4/erp_valid_masks.npz",
}
CAMERAS = ("cam2", "cam4")
CONFIDENCE = 0.15
MAX_BOX_AREA_FRACTION = 0.1


def freeze():
    EXPERIMENT.mkdir(parents=True, exist_ok=False)
    images = []
    for dataset in ("data3", "data4"):
        by_camera = {
            camera: {p.name: p for p in (RIDING / dataset / "erp" / camera).glob("*.png")}
            for camera in CAMERAS
        }
        if by_camera["cam2"].keys() != by_camera["cam4"].keys():
            raise ValueError(f"unsynchronized_source_images:{dataset}")
        names = sorted(by_camera["cam2"])
        if len(names) < 20:
            raise ValueError(f"too_few_source_pairs:{dataset}")
        # One pair per temporal stratum; the selection uses no model output.
        selected = [names[int((i + 0.5) * len(names) / 20)] for i in range(20)]
        if len(set(selected)) != 20:
            raise ValueError("duplicate_selected_pair")
        for name in selected:
            for camera in CAMERAS:
                path = by_camera[camera][name].resolve()
                images.append({
                    "image_id": f"{dataset}_{camera}_{Path(name).stem}",
                    "dataset": dataset,
                    "source_frame": Path(name).stem,
                    "camera": camera,
                    "source_png": str(path),
                    "reason": "时间轴第 %d/20 段的中点；两机同步保留" % (selected.index(name) + 1),
                    "calibration": "20260320" if dataset == "data3" else "20260311",
                    "valid_mask_resource": str(VALID[dataset]),
                })
    manifest = {"schema": "real_riding_selection_v1", "selection_method": "20 equal temporal strata per dataset, paired cameras, independent of model predictions", "images": images}
    (EXPERIMENT / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # A compact original-image index for visual checking; no model result is shown.
    sheet = np.full((10 * 120, 8 * 240, 3), 255, np.uint8)
    for index, row in enumerate(images):
        image = cv2.imread(row["source_png"], cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2880, 5760):
            raise ValueError(f"invalid_source_png:{row['image_id']}")
        thumb = cv2.resize(image, (240, 120), interpolation=cv2.INTER_AREA)
        y, x = divmod(index, 8)
        sheet[y * 120:(y + 1) * 120, x * 240:(x + 1) * 240] = thumb
        cv2.putText(sheet, row["image_id"], (x * 240 + 4, y * 120 + 16), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(EXPERIMENT / "selection_contact_sheet.jpg"), sheet)
    print(f"Frozen {len(images)} original images in {EXPERIMENT / 'selection.json'}")


def detect(device):
    from ultralytics import YOLO

    manifest = json.loads((EXPERIMENT / "selection.json").read_text())
    detector = YOLO(str(ENGINE))
    targets = []
    image_reports = []
    valid_masks = {}
    for dataset in VALID:
        with np.load(VALID[dataset], allow_pickle=False) as data:
            valid_masks[dataset] = {camera: data[f"valid_masks_{camera}"].copy() for camera in CAMERAS}
    mask_root = EXPERIMENT / "masks"
    mask_root.mkdir(exist_ok=False)
    for index, row in enumerate(manifest["images"], 1):
        image = cv2.imread(row["source_png"], cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        scale = min(1., 1600 / max(height, width))
        small = cv2.resize(image, (round(width * scale), round(height * scale)))
        result = detector.predict(small, imgsz=(640, 1280), conf=.05, classes=[0], max_det=50,
                                  retina_masks=False, verbose=False, device=device)[0]
        report = {"image_id": row["image_id"], "raw_detections": len(result.boxes), "accepted": 0,
                  "rejected": []}
        for raw_index in range(len(result.boxes)):
            score = float(result.boxes.conf[raw_index])
            box = result.boxes.xyxy[raw_index].cpu().numpy() / scale
            area_fraction = float(max(0., box[2] - box[0]) * max(0., box[3] - box[1]) / (width * height))
            reason = None
            if score < CONFIDENCE:
                reason = "confidence_below_0.15"
            elif area_fraction > MAX_BOX_AREA_FRACTION:
                reason = "box_exceeds_10pct_erp"
            elif result.masks is None or raw_index >= len(result.masks.xy):
                reason = "missing_segmentation_mask"
            if reason:
                report["rejected"].append({"raw_index": raw_index, "score": score, "reason": reason})
                continue
            polygon = result.masks.xy[raw_index]
            mask = np.zeros((height, width), np.uint8)
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [np.rint(polygon / scale).astype(np.int32)], 255)
            mask[~valid_masks[row["dataset"]][row["camera"]]] = 0
            ys, xs = np.nonzero(mask)
            if len(xs) == 0 or xs.max() <= xs.min() or ys.max() <= ys.min():
                report["rejected"].append({"raw_index": raw_index, "score": score, "reason": "empty_or_degenerate_valid_mask"})
                continue
            target_id = f"{row['image_id']}_p{raw_index:02d}"
            mask_path = mask_root / f"{target_id}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"mask_write_failed:{target_id}")
            targets.append({"target_id": target_id, "image_id": row["image_id"], "raw_index": raw_index,
                            "mask": str(mask_path.relative_to(EXPERIMENT)), "score": score,
                            "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                            "foreground_pixels": len(xs)})
            report["accepted"] += 1
        image_reports.append(report)
        print(f"{index}/{len(manifest['images'])} {row['image_id']}: {report['accepted']} targets", flush=True)
    frozen = {"schema": "real_riding_targets_v1", "selection": "selection.json", "detector": {
        "engine": str(ENGINE), "name": "YOLO26s-seg FP32 TensorRT", "resized_long_side": 1600,
        "imgsz": [640, 1280], "predict_confidence": .05, "target_confidence": CONFIDENCE,
        "class": 0, "max_det": 50, "max_box_area_fraction": MAX_BOX_AREA_FRACTION,
        "mask_method": "YOLO polygon projected to original ERP, clipped by dataset/camera physical valid mask",
    }, "images": image_reports, "targets": targets}
    (EXPERIMENT / "targets.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(targets)} person targets in {EXPERIMENT / 'targets.json'}")


def preflight():
    """Freeze native-crop eligibility before either MAMMA model is loaded."""
    from mamma_ext.data import native_plan, periodic_mask_box

    source = json.loads((EXPERIMENT / "targets.json").read_text())
    if source["schema"] != "real_riding_targets_v1":
        raise ValueError("invalid_detected_targets")
    roi = np.array([0, 0, 5760, 2880])
    size = np.array([5760, 2880])
    ready, rejected = [], []
    for row in source["targets"]:
        mask = cv2.imread(str(EXPERIMENT / row["mask"]), cv2.IMREAD_GRAYSCALE)
        try:
            box = periodic_mask_box(mask, roi[:2], 5760)
            _, _, _, rect = native_plan(
                box, roi, size, size, periodic_x=True, vertical_pad_black=True
            )
        except ValueError as error:
            rejected.append({"target_id": row["target_id"], "reason": str(error)})
        else:
            ready.append({**row, "vertical_padding_top_bottom": [
                max(0, -int(rect[1])), max(0, int(rect[3]) - 2880)
            ]})
    result = {"schema": "real_riding_ready_targets_v1", "source": "targets.json",
              "rule": "existing periodic mask-derived 384x512 crop requires complete source filter support",
              "targets": ready, "rejected": rejected}
    (EXPERIMENT / "targets_ready.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(ready)} inference targets; {len(rejected)} cannot support native crop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "detect", "preflight"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "detect":
        detect(args.device)
    else:
        preflight()


if __name__ == "__main__":
    main()
