"""CPU child: one fixed, trusted MAMMA deployment checkpoint -> tensor-only state.

This uses pickle for the original local file exactly as the deployed PyTorch path
requires. Process isolation limits ambient side effects; it does not make pickle safe.
"""

import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/yy/Repos/mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt")
OUT = ROOT / "outputs/experiments/mamma_original_baseline_pilot_20261015"
STATE = OUT / "tensor_state.pt"
PARTIAL = OUT / "tensor_state.pt.partial"
META = OUT / "weight_metadata.json"
FAILED = OUT / "checkpoint_export_failure.json"


def main():
    started = time.monotonic()
    if STATE.exists() or PARTIAL.exists():
        raise RuntimeError("owned_tensor_export_already_exists")
    if SOURCE.stat().st_size != 1621536317:
        raise RuntimeError("trusted_fixed_source_size_changed")
    report = dict(
        source=str(SOURCE),
        source_size_bytes=SOURCE.stat().st_size,
        source_repo_head="588492f18876e2ed6888b2d26929047cb6b575e7",
        source_registry="inference/assets.py:ma_2d_checkpoint",
        deployment_loader="/home/yy/Repos/riding-data/run_model_comparison.py:203-205",
        python=sys.executable,
        torch=torch.__version__,
        method="one CPU subprocess torch.load(weights_only=False,map_location=cpu) fixed trusted local file; tensor-only torch.save export",
        pickle_risk="child process bounds side effects but does not eliminate pickle execution risk",
    )
    try:
        obj = torch.load(SOURCE, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or not isinstance(obj.get("state_dict"), dict):
            raise TypeError("checkpoint_missing_state_dict")
        state = obj["state_dict"]
        if not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) and v.device.type == "cpu"
            for k, v in state.items()
        ):
            raise TypeError("state_dict_not_cpu_string_tensor_map")
        report["top_level_keys"] = list(obj)
        report["epoch"] = (
            obj.get("epoch") if isinstance(obj.get("epoch"), (int, float, str)) else None
        )
        report["global_step"] = (
            obj.get("global_step")
            if isinstance(obj.get("global_step"), (int, float, str))
            else None
        )
        report["lightning_version"] = (
            obj.get("pytorch-lightning_version")
            if isinstance(obj.get("pytorch-lightning_version"), str)
            else None
        )
        hp = obj.get("hyper_parameters")
        report["hyper_parameters_type"] = type(hp).__name__ if hp is not None else None
        if isinstance(hp, dict):
            report["hyper_parameter_keys"] = [str(k) for k in hp]
        report["state_key_count"] = len(state)
        report["state_keys_shapes_dtypes"] = {
            k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in state.items()
        }
        report["state_tensor_bytes"] = sum(v.numel() * v.element_size() for v in state.values())
        torch.save(dict(state), PARTIAL)
        os.replace(PARTIAL, STATE)
        report["tensor_export"] = str(STATE)
        report["tensor_export_size_bytes"] = STATE.stat().st_size
        report["status"] = "tensor_state_exported"
        report["elapsed_seconds"] = time.monotonic() - started
        META.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "state_keys_shapes_dtypes"},
                indent=2,
                default=str,
            )
        )
    except Exception as exc:
        FAILED.write_text(
            json.dumps(
                dict(
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc)[:2000],
                    elapsed_seconds=time.monotonic() - started,
                    partial_file_exists=PARTIAL.exists(),
                ),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
