"""Build a MAMMA-vertex fixture and a local variant of pano-stitch's LM worker.

The pano-stitch source and its delivered binary remain untouched.  The only
worker source edits are the two DenseKP-specific 138-point request bounds;
the numerical solver and its priors are compiled from the read-only source.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PANO = Path("/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch")
PREDICTION = ROOT / "outputs/experiments/real_riding_manual_review_20261024/predictions_original.npz"
BASE = PANO / "artifacts/human_fitting_delivery/fitting/static_fixture"
MODEL = PANO / "tools/smpl_models/smplx/SMPLX_NEUTRAL.npz"


def _metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _save_metadata(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in sorted(values.items())))


def prepare_fixture(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with np.load(PREDICTION, allow_pickle=False) as data:
        if str(data["template_id"]) != "MAMMA_verts_512_ordered_v1":
            raise ValueError("unexpected MAMMA template")
        ids = data["ordered_vertex_ids"].copy()
    if ids.shape != (512,) or len(np.unique(ids)) != 512 or ids.min() < 0:
        raise ValueError("invalid ordered MAMMA vertex IDs")
    with np.load(MODEL, allow_pickle=False) as model:
        if ids.max() >= len(model["v_template"]):
            raise ValueError("MAMMA vertex exceeds SMPL-X topology")
        # The delivered C++ fixture is validated against this full neutral
        # model before selecting new rows.  It therefore keeps the original
        # joint, pose, shape and projection semantics.
        old_ids = np.fromfile(BASE / "densekp_support_vertex_ids.i32", dtype="<i4")
        old_base = np.fromfile(BASE / "model_cam2_selected_base_v_shaped.f32", dtype="<f4").reshape(204, 3)
        old_beta = np.fromfile(BASE / "model_cam2_selected_beta_shapedirs.f32", dtype="<f4").reshape(204, 3, 10)
        old_pose = np.fromfile(BASE / "model_cam2_selected_posedirs.f32", dtype="<f4").reshape(486, 204, 3)
        old_weights = np.fromfile(BASE / "model_cam2_selected_lbs_weights.f32", dtype="<f4").reshape(204, 55)
        if not (
            np.array_equal(old_base, model["v_template"][old_ids])
            and np.allclose(old_beta, model["shapedirs"][old_ids, :, :10], atol=1e-8, rtol=0)
            and np.array_equal(old_pose, model["posedirs"][old_ids].transpose(2, 0, 1))
            and np.array_equal(old_weights, model["weights"][old_ids])
        ):
            raise ValueError("neutral SMPL-X source differs from delivered fitting fixture")
        selected = {
            "selected_base_v_shaped": model["v_template"][ids],
            "selected_beta_shapedirs": model["shapedirs"][ids, :, :10],
            "selected_posedirs": model["posedirs"][ids].transpose(2, 0, 1),
            "selected_lbs_weights": model["weights"][ids],
        }
        output.mkdir(parents=True)
        common = (
            "pipeline_meta.txt", "stage1_fixture_meta.txt", "model_fixture_meta.txt",
            "model_base_joints.f32", "model_d_joints_d_beta.f32",
            "model_parents.i64",
        )
        for name in common:
            shutil.copyfile(BASE / name, output / name)
        for camera in ("cam2", "cam4"):
            for name, array in selected.items():
                np.ascontiguousarray(array, dtype="<f4").tofile(output / f"model_{camera}_{name}.f32")
        np.ascontiguousarray(ids, dtype="<i4").tofile(output / "densekp_support_vertex_ids.i32")
        # The solver's sparse observation operator is generic.  An identity
        # operator makes observation row i the exact SMPL-X vertex ids[i].
        np.eye(512, dtype="<f4").tofile(output / "densekp_regressor_138x204.f32")
        meta = _metadata(output / "model_fixture_meta.txt")
        meta.update({
            "observation_mode": "mamma_512_vertex_identity",
            "densekp_support_count": "512",
            "densekp_row_count": "512",
        })
        for camera in ("cam2", "cam4"):
            meta[f"{camera}_count"] = "512"
            meta[f"{camera}_support_count"] = "512"
            meta[f"{camera}_observation_count"] = "512"
            meta[f"{camera}_alpha"] = repr(1.0 / 512)
        _save_metadata(output / "model_fixture_meta.txt", meta)
        (output / "mamma_template_vertex_ids.i32").write_bytes(
            (output / "densekp_support_vertex_ids.i32").read_bytes()
        )


def prepare_source(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    source = (PANO / "cpp/staged_lm/apps/persistent_staged_lm_worker_main.cpp").read_text()
    replacements = {
        "count > 138": "count > 512",
        "id >= 138": "id >= 512",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"unexpected external worker contract: {old}")
        source = source.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    prepare_fixture(root / "fixture")
    source = prepare_source(root / "generated/persistent_mamma_512_worker.cpp")
    if args.build:
        build = root / "build"
        subprocess.run([
            "/usr/bin/cmake", "-S", str(Path(__file__).parent / "cpp"), "-B", str(build),
            f"-DPANO_CPP={PANO / 'cpp/staged_lm'}", f"-DADAPTED_SOURCE={source}",
            "-DSTAGED_LM_ENABLE_CUDA=OFF", "-DSTAGED_LM_MARCH_NATIVE=OFF",
        ], check=True)
        subprocess.run(["/usr/bin/cmake", "--build", str(build), "--target", "mamma_512_worker", "-j", "2"], check=True)
    print(root)


if __name__ == "__main__":
    main()
