"""Compare frozen original MAMMA stitching with its SMPL-X fitting variant.

Run from the mamma_retrain project.  The first two processed frames are the
first matched data3 and data4 frames; --limit-pairs 2 is the fixed pilot.
The same output can then be resumed to 40 pairs without refitting the pilot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PANO = Path("/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch")
OLD = ROOT / "outputs/experiments/real_riding_stitch_threshold085_20261025"
DEFAULT = ROOT / "outputs/experiments/original_mamma_fitting_20261025"
CAMERAS = ("cam2", "cam4")


def _external_paths() -> None:
    for path in (
        PANO / "src", PANO / "tools/4D-Humans", PANO / "tools/SingleFitting",
        PANO / "dist/human_fitting_delivery/vendor",
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))
    sys.path.insert(0, str(ROOT / "scripts"))


def _periodic_distance(a: np.ndarray, b: np.ndarray, width: int = 5760) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + width / 2) % width - width / 2)


def _erp_rotate_to_erp(points: np.ndarray) -> np.ndarray:
    from lib_pano.fitting_io.densekp_delivery_io import (
        ERP_TO_ERP_ROTATE, _erp_to_ray, _ray_to_erp,
    )

    rays = _erp_to_ray(np.asarray(points, dtype=np.float64), 5760, 2880)
    return _ray_to_erp(rays @ ERP_TO_ERP_ROTATE, 5760, 2880)


def verify_rotation(points: np.ndarray) -> float:
    from lib_pano.fitting_io.densekp_delivery_io import erp_to_erp_rotate

    restored = _erp_rotate_to_erp(erp_to_erp_rotate(points, 5760, 2880))
    error = np.hypot(_periodic_distance(restored[:, 0], points[:, 0]),
                     restored[:, 1] - points[:, 1])
    maximum = float(np.max(error))
    if maximum > 0.002:
        raise ValueError(f"ERP sphere rotation round trip: {maximum} px")
    return maximum


def _worker_class():
    # Reuse the production transport/response parser while changing only the
    # 138-row protocol guard.  The matching C++ worker is built locally.
    import subprocess
    import threading

    from lib_pano.human_fitting.persistent_cpp_worker import (
        PersistentCppFittingWorker, WorkerFatalError, _Reader,
        PROTOCOL_VERSION,
    )

    class MammaWorker(PersistentCppFittingWorker):
        def start(self):
            if self.process is not None:
                if self.process.poll() is not None:
                    raise WorkerFatalError("MAMMA worker exited")
                return self
            executable = self.config.executable.resolve()
            fixture = self.config.static_fixture.resolve()
            if not executable.is_file() or not fixture.is_dir():
                raise FileNotFoundError((executable, fixture))
            self.process = subprocess.Popen(
                [str(executable), "--static-fixture", str(fixture)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
            self.start_count += 1
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, name="mamma-512-worker-stderr", daemon=True,
            )
            self._stderr_thread.start()
            try:
                ready = _Reader(self._receive_response())
                loads = ready.unpack("I")
                actual = (ready.unpack("I"), ready.string(), ready.unpack("I"),
                          ready.unpack("I"), ready.unpack("I"), Path(ready.string()).resolve())
                self.worker_startup_ms = ready.unpack("d")
                self.static_model_load_ms = ready.unpack("d")
                expected = (PROTOCOL_VERSION, "cpu", 75, 10, 512, fixture)
                if actual != expected or loads != 1 or ready.offset != len(ready.payload):
                    raise WorkerFatalError(f"MAMMA worker READY mismatch: {actual} != {expected}")
                self.static_model_load_count = loads
            except Exception:
                self._mark_failed()
                raise
            return self

        @staticmethod
        def _request_arrays(person, initial):
            initial.validate()
            ids = np.ascontiguousarray(person.point_ids[person.valid], dtype="<i4")
            pairs = np.ascontiguousarray(person.pairs[person.valid], dtype="<f4")
            if not 0 < len(ids) <= 512 or ids.min() < 0 or ids.max() >= 512:
                raise ValueError("MAMMA worker requires 1..512 template row IDs")
            return len(ids), [
                np.asarray(initial.frame_parameters, dtype="<f4"),
                np.asarray(initial.beta, dtype="<f4"),
                np.asarray(initial.cam2_left_hand_pose, dtype="<f4"),
                np.asarray(initial.cam2_right_hand_pose, dtype="<f4"),
                np.asarray(initial.cam4_left_hand_pose, dtype="<f4"),
                np.asarray(initial.cam4_right_hand_pose, dtype="<f4"),
                np.asarray([initial.cam2_scale, initial.cam4_scale], dtype="<f4"),
                ids, pairs[:, :2].copy(), pairs[:, 2:].copy(),
            ]

    return MammaWorker


def _order_pairs(pair_frames, old_index):
    old_map = {(row["dataset"], row["source_frame"]): row for row in old_index["pairs"]}
    if set(old_map) != {key for key, _ in pair_frames}:
        raise ValueError("old direct branch pair list differs from frozen selection")
    first = []
    for dataset in ("data3", "data4"):
        first.append(next(key for key, _ in pair_frames
                          if key[0] == dataset and old_map[key]["branches"]["original_050"]["person_pairs"]))
    return sorted(pair_frames, key=lambda pair: (pair[0] not in first,
                                                  first.index(pair[0]) if pair[0] in first else 0,
                                                  pair[0]))


def _frozen_original_inputs(review: Path):
    """Read only the frozen original-model prediction and its exact targets."""
    selection = json.loads((review / "selection.json").read_text())
    target_file = json.loads((review / "targets.json").read_text())
    with np.load(review / "predictions_original.npz", allow_pickle=False) as source:
        prediction = {key: source[key].copy() for key in source.files}
    images, targets = selection["images"], target_file["targets"]
    ids = [row["target_id"] for row in targets]
    if len(images) != 80 or len(targets) != 319 or len(set(ids)) != 319:
        raise ValueError("original frozen input count changed")
    if set(ids) != set(prediction["target_ids"].tolist()):
        raise ValueError("original prediction target identity differs")
    xy, prob = prediction["xy_full_erp"], prediction["visibility_probability"]
    if (str(prediction["template_id"]) != "MAMMA_verts_512_ordered_v1"
            or prediction["ordered_vertex_ids"].shape != (512,)
            or prediction["input_affine_erp_to_model"].shape != (319, 2, 3)
            or xy.shape != (319, 512, 2) or prob.shape != (319, 512)
            or not np.isfinite(xy).all() or not np.isfinite(prob).all()
            or np.any((prob < 0) | (prob > 1))):
        raise ValueError("original prediction contract differs")
    prediction_index = {target_id: index for index, target_id in
                        enumerate(prediction["target_ids"].tolist())}
    by_image = defaultdict(list)
    for row in targets:
        by_image[row["image_id"]].append((prediction_index[row["target_id"]], row))
    pairs = defaultdict(dict)
    for image in images:
        key = image["dataset"], image["source_frame"]
        camera = image["camera"]
        if camera not in CAMERAS or camera in pairs[key]:
            raise ValueError("original frozen image camera duplicated")
        pairs[key][camera] = image
    if len(pairs) != 40 or any(set(pair) != set(CAMERAS) for pair in pairs.values()):
        raise ValueError("original frozen paired image coverage differs")
    return sorted(pairs.items()), by_image, prediction


def _fitting_person(decision, edge):
    from lib_pano.fitting_io.densekp_delivery_io import erp_to_erp_rotate
    from lib_pano.human_fitting.matched_person_tracks import MatchedPerson

    i, j = edge["i"], edge["j"]
    left = decision["observations"][0][i]
    right = decision["observations"][1][j]
    keep = decision["keep"][i, j]
    ids = np.flatnonzero(keep).astype(np.int32)
    p2 = decision["arrays"][0]["landmarks"][i][ids, :2]
    p4 = decision["arrays"][1]["landmarks"][j][ids, :2]
    if len(ids) != edge["common_points"] or len(ids) == 0:
        raise ValueError("D pairing common-point identity changed")
    verify_rotation(p2)
    verify_rotation(p4)
    normal = MatchedPerson(
        person_id=i,
        bbox_cam2=np.asarray(left["mamma"]["box"], np.float32),
        bbox_cam4=np.asarray(right["mamma"]["box"], np.float32),
        pairs=np.column_stack((p2, p4)).astype(np.float32),
        point_ids=ids,
        valid=np.ones(len(ids), bool),
    )
    rotated = MatchedPerson(
        person_id=i, bbox_cam2=normal.bbox_cam2, bbox_cam4=normal.bbox_cam4,
        pairs=np.column_stack((erp_to_erp_rotate(p2, 5760, 2880),
                               erp_to_erp_rotate(p4, 5760, 2880))).astype(np.float32),
        point_ids=ids, valid=normal.valid.copy(),
    )
    return normal, rotated, left, right, keep


def _initial_state(hmr, normal, rotated, images):
    from lib_pano.human_fitting.matched_person_tracks import choose_best_view
    from lib_pano.human_fitting.hmr2_staged_init import (
        convert_hmr2_to_staged_initialization, spherical_translation_initialization,
        to_smplx_camera_initializations,
    )
    from lib_pano.fitting_io.densekp_runtime_export import person_initial_state_from_smplx
    from dataclasses import replace

    selected = choose_best_view(normal)
    prediction = hmr.infer(images[selected.camera], selected.bbox_xyxy.reshape(1, 4))[0]
    focal = float(hmr.model_cfg.EXTRA.FOCAL_LENGTH) / float(hmr.model_cfg.MODEL.IMAGE_SIZE) * 5760.0
    staged = convert_hmr2_to_staged_initialization(
        prediction, source_camera=selected.camera,
        bboxes={"cam2": normal.bbox_cam2, "cam4": normal.bbox_cam4},
        focal_length=focal,
    )
    # The legacy HMR2 pinhole translation collapses under this directional
    # ERP model.  Use the production observation-derived spherical rule.
    t2, t4 = spherical_translation_initialization(
        rotated.pairs[:, :2], rotated.pairs[:, 2:], rotated.valid, 5760, 2880,
        bbox_cam2=normal.bbox_cam2, bbox_cam4=normal.bbox_cam4,
    )
    staged = replace(staged, translation_cam2=t2, translation_cam4=t4)
    state = person_initial_state_from_smplx(*to_smplx_camera_initializations(staged))
    return state, selected.camera


def _image_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("result image outside fixed root")
    return resolved


def run(args) -> None:
    _external_paths()
    import real_riding_stitch_compare as direct
    from lib_pano.human3d.hmr2_initializer import HMR2Initializer
    from lib_pano.human_fitting.persistent_cpp_worker import PersistentCppWorkerConfig

    pair_frames, by_image, prediction = _frozen_original_inputs(direct.REVIEW)
    fixture = args.adapter / "fixture"
    saved_ids = np.fromfile(fixture / "mamma_template_vertex_ids.i32", dtype="<i4")
    if not np.array_equal(saved_ids, prediction["ordered_vertex_ids"]):
        raise ValueError("fixture does not match original MAMMA point order")
    if not direct.MODEL_WEIGHTS["original"].is_file():
        raise FileNotFoundError(direct.MODEL_WEIGHTS["original"])
    original_index = json.loads((OLD / "index.json").read_text())
    pair_frames = _order_pairs(pair_frames, original_index)
    pair_current, gate_warp, NativeComposite, render_mamma, load_masks = direct.external_modules(
        args.riding_data, args.deployment_source, args.densekp_vendor,
    )
    output = args.output.resolve()
    index_path = output / "index.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if args.resume:
        result = json.loads(index_path.read_text())
        if result["schema"] != "original_mamma_fitting_comparison_v1":
            raise ValueError("wrong fitting result schema")
    else:
        output.mkdir(parents=True)
        result = {
            "schema": "original_mamma_fitting_comparison_v1",
            "model": "original", "threshold": 0.5,
            "template_id": str(prediction["template_id"]),
            "source_prediction": str(direct.REVIEW / "predictions_original.npz"),
            "baseline_index": str(OLD / "index.json"),
            "worker": str(args.adapter / "build/mamma_512_worker"),
            "fixture": str(fixture), "hmr2_checkpoint": str(args.hmr2_checkpoint),
            "gpu": args.gpu, "pairs": [],
        }
    done = {(row["dataset"], row["source_frame"]) for row in result["pairs"]}
    if done != {key for key, _ in pair_frames[:len(done)]}:
        raise ValueError("resume prefix differs from frozen processing order")
    worker_type = _worker_class()
    worker = worker_type(PersistentCppWorkerConfig(
        executable=args.adapter / "build/mamma_512_worker",
        static_fixture=fixture, timeout_seconds=args.fit_timeout,
    )).start()
    hmr = HMR2Initializer(args.hmr2_checkpoint, device=args.gpu)
    masks_by_calibration = {}
    started = time.monotonic()
    try:
        for key, source in pair_frames[len(done):args.limit_pairs]:
            dataset, frame_name = key
            old_row = next(row for row in original_index["pairs"]
                           if (row["dataset"], row["source_frame"]) == key)
            resources = {source[c]["valid_mask_resource"] for c in CAMERAS}
            calibrations = {source[c]["calibration"] for c in CAMERAS}
            if len(resources) != 1 or calibrations != {old_row["calibration"]}:
                raise ValueError(f"calibration mismatch: {key}")
            if (dataset, old_row["calibration"]) not in (("data3", "20260320"), ("data4", "20260311")):
                raise ValueError(f"unexpected dataset calibration: {key}")
            calibration = old_row["calibration"]
            if calibration not in masks_by_calibration:
                valid, small = load_masks(Path(resources.pop()), (1920, 960))
                masks_by_calibration[calibration] = valid, small, NativeComposite(valid)
            valid, small, composite = masks_by_calibration[calibration]
            camera_data = {cam: direct.load_camera_data(source[cam], by_image[source[cam]["image_id"]])
                           for cam in CAMERAS}
            obs = [direct.observations(camera_data[cam][1], prediction, 0.5) for cam in CAMERAS]
            decision = pair_current(obs, valid)
            frozen_pairs = old_row["branches"]["original_050"]["person_pairs"]
            actual_pairs = [{"cam2_target": decision["observations"][0][e["i"]]["target_id"],
                             "cam4_target": decision["observations"][1][e["j"]]["target_id"],
                             "common_points": e["common_points"]} for e in decision["pairs"]]
            if actual_pairs != frozen_pairs:
                raise ValueError(f"D pairing differs from frozen baseline: {key}")
            groups, masks, fits, admissions = [], [], [], []
            for edge in decision["pairs"]:
                normal, rotated, left, right, keep = _fitting_person(decision, edge)
                raw = normal.pairs.copy()
                identity = (left["target_id"], right["target_id"])
                t0 = time.monotonic()
                try:
                    initial, source_camera = _initial_state(
                        hmr, normal, rotated,
                        {cam: camera_data[cam][0] for cam in CAMERAS},
                    )
                    init_s = time.monotonic() - t0
                    fitted = worker.fit(rotated, initial)
                    if not fitted.success or fitted.optimized_pairs.shape != raw.shape:
                        raise RuntimeError(f"fitting unsuccessful: {fitted.status}")
                    projected = np.column_stack((
                        _erp_rotate_to_erp(fitted.optimized_pairs[:, :2]),
                        _erp_rotate_to_erp(fitted.optimized_pairs[:, 2:]),
                    ))
                    if not np.isfinite(projected).all():
                        raise ValueError("fitting returned nonfinite ERP coordinates")
                    displacement = np.hypot(_periodic_distance(projected[:, 0], raw[:, 0]),
                                            projected[:, 1] - raw[:, 1])
                    fitted_group = np.column_stack((projected[:, 2:], projected[:, :2])).astype(np.float32)
                    fit_record = {"cam2_target": identity[0], "cam4_target": identity[1],
                                  "status": fitted.status, "success": True,
                                  "initialization_camera": source_camera,
                                  "point_ids": normal.point_ids.astype(int).tolist(),
                                  "common_points": len(normal.point_ids),
                                  "initialization_seconds": round(init_s, 3),
                                  "fitting_seconds": round(time.monotonic() - t0 - init_s, 3),
                                  "initial_objective": fitted.initial_objective,
                                  "final_objective": fitted.final_objective,
                                  "median_displacement_px": float(np.median(displacement)),
                                  "max_displacement_px": float(np.max(displacement)),
                                  "projection_extent_ratio": fitted.projection_extent_ratio}
                except Exception as error:
                    fitted_group = np.column_stack((raw[:, 2:], raw[:, :2])).astype(np.float32)
                    fit_record = {"cam2_target": identity[0], "cam4_target": identity[1],
                                  "status": "FITTING_FAILED", "success": False,
                                  "reason": f"{type(error).__name__}: {error}",
                                  "point_ids": normal.point_ids.astype(int).tolist(),
                                  "common_points": len(normal.point_ids),
                                  "elapsed_seconds": round(time.monotonic() - t0, 3)}
                reason, details = gate_warp(fitted_group, "cam4_to_cam2", small)
                admissions.append({"cam2_target": identity[0], "cam4_target": identity[1],
                                   "gate": reason, "details": details,
                                   "point_ids": normal.point_ids.astype(int).tolist()})
                if reason == "admit":
                    groups.append(fitted_group)
                    masks.append({"cam2": left["mask"], "cam4": right["mask"]})
                fits.append(fit_record)
            if groups:
                image, warped, failures, ownership = render_mamma(
                    camera_data["cam2"][0], camera_data["cam4"][0], valid,
                    groups, masks, small, composite,
                )
            else:
                image = composite.combine(camera_data["cam2"][0], camera_data["cam4"][0], None)
                warped, failures, ownership = [], [], []
            if image.shape != (2880, 5760, 3):
                raise ValueError("fitted stitch output shape changed")
            dest = output / "frames" / f"{dataset}_{frame_name}"
            dest.mkdir(parents=True, exist_ok=False)
            image_path = dest / "fitted_050.png"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(image_path)
            preview = output / "previews" / f"{dataset}_{frame_name}"
            preview.mkdir(parents=True, exist_ok=False)
            preview_path = preview / "fitted_050.png"
            if not cv2.imwrite(str(preview_path), cv2.resize(image, (1440, 720), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(preview_path)
            base = old_row["branches"]["original_050"]
            base_file = _image_path(OLD, base["image"])
            if not base_file.is_file():
                raise FileNotFoundError(base_file)
            row = {"dataset": dataset, "source_frame": frame_name,
                   "calibration": calibration,
                   "cam2_image": source["cam2"]["source_png"],
                   "cam4_image": source["cam4"]["source_png"],
                   "baseline_image": str(base_file),
                   "baseline_preview": str(_image_path(OLD, base["preview"])),
                   "fitted_image": str(image_path.relative_to(output)),
                   "fitted_preview": str(preview_path.relative_to(output)),
                   "person_pairs": actual_pairs, "fitting": fits,
                   "warp_admission": admissions,
                   "warped": warped, "warp_failures": failures,
                   "ownership_fallback": ownership,
                   "status": "WARPED" if warped and not failures else
                             "PARTIAL WARP" if warped else
                             "WARP REJECTED" if actual_pairs else "NO MATCH"}
            result["pairs"].append(row)
            temporary = index_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(index_path)
            print(f"{len(result['pairs'])}/40 {dataset}/{frame_name} "
                  f"matched={len(actual_pairs)} fitted={sum(x['success'] for x in fits)} "
                  f"status={row['status']}", flush=True)
    finally:
        worker.close()
    print(f"processed this invocation in {time.monotonic()-started:.1f}s; {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT / "adapter")
    parser.add_argument("--output", type=Path, default=DEFAULT / "comparison")
    parser.add_argument("--hmr2-checkpoint", type=Path,
                        default=PANO / "artifacts/human_fitting_delivery/hmr2/checkpoints/epoch=35-step=1000000.ckpt")
    parser.add_argument("--riding-data", type=Path, default=Path("/home/yy/Repos/riding-data"))
    parser.add_argument("--deployment-source", type=Path, default=Path("/home/yy/Repos/mamma-deploy/src"))
    parser.add_argument("--densekp-vendor", type=Path,
                        default=PANO / "dist/human_fitting_delivery/vendor")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--fit-timeout", type=float, default=60)
    parser.add_argument("--limit-pairs", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit_pairs <= 40 or args.fit_timeout <= 0:
        parser.error("limit-pairs must be 1..40 and fit-timeout positive")
    run(args)


if __name__ == "__main__":
    main()
