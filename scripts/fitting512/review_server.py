"""Serve the fixed original-MAMMA versus original-MAMMA-plus-fitting images."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "outputs/experiments/original_mamma_fitting_20261025/comparison_gpu_trial2"
BASELINE = ROOT / "outputs/experiments/real_riding_stitch_threshold085_20261025"
SOURCE = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
sys.path.insert(0, str(ROOT / "scripts"))
from real_stitch_review_server import handler  # noqa: E402


def permitted(experiment: Path):
    result = json.loads((experiment / "index.json").read_text())
    baseline = json.loads((BASELINE / "index.json").read_text())
    selection = json.loads((SOURCE / "selection.json").read_text())
    targets = json.loads((SOURCE / "targets.json").read_text())
    if result["schema"] != "original_mamma_fitting_comparison_v1" or len(result["pairs"]) != 40:
        raise ValueError("fitting comparison is not complete")
    baseline_rows = {(r["dataset"], r["source_frame"]): r for r in baseline["pairs"]}
    sources = {(r["dataset"], r["source_frame"], r["camera"]): Path(r["source_png"]).resolve()
               for r in selection["images"]}
    files, pairs = {}, []
    for number, row in enumerate(result["pairs"]):
        key = row["dataset"], row["source_frame"]
        old = baseline_rows[key]
        for camera in ("cam2", "cam4"):
            files[f"/source/{number}/{camera}"] = sources[(*key, camera)]
        for branch, detail in (("original_050", old["branches"]["original_050"]),
                               ("fitted_050", row)):
            for route, field in (("frame", "image"), ("preview", "preview")):
                if branch == "original_050":
                    path = (BASELINE / detail[field]).resolve()
                    if not path.is_relative_to(BASELINE.resolve()):
                        raise ValueError("baseline image outside fixed result")
                else:
                    name = "fitted_image" if route == "frame" else "fitted_preview"
                    path = (experiment / detail[name]).resolve()
                    if not path.is_relative_to(experiment.resolve()):
                        raise ValueError("fitting image outside fixed result")
                files[f"/{route}/{number}/{branch}"] = path
        fitted = {name: row[name] for name in
                  ("status", "person_pairs", "warp_admission", "warped", "warp_failures", "fitting")}
        pairs.append({"dataset": key[0], "source_frame": key[1],
                      "calibration": row["calibration"],
                      "branches": {"original_050": old["branches"]["original_050"],
                                   "fitted_050": fitted}})
    if len(set((r["dataset"], r["source_frame"]) for r in pairs)) != 40:
        raise ValueError("duplicate comparison frame")
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("missing fixed source or stitch image")
    index = {
        "schema": "original_mamma_fitting_comparison_v1",
        "review_set_id": "real_riding_40_pairs_original050_vs_smplx512_fitting050_v1",
        "description": "固定 40 对真实 ERP；同一未微调 MAMMA 与阈值 0.5，右侧只增加逐帧人体 fitting。人物配对及输入点编号保持一致；两栏使用相同演示合图链。",
        "input_selection": str(SOURCE / "selection.json"),
        "prediction_files": {"original": result["source_prediction"]},
        "model_weights": {"original": str(ROOT / "outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt")},
        "branches": {
            "original_050": {"model": "original", "threshold": 0.5,
                             "label": "原 MAMMA · 阈值 0.5"},
            "fitted_050": {"model": "original+smplx512_fitting", "threshold": 0.5,
                           "label": "原 MAMMA＋人体 fitting · 阈值 0.5"},
        },
        "pairs": pairs,
    }
    boxes = {row["target_id"]: row["box_xyxy"] for row in targets["targets"]}
    return index, files, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18873)
    args = parser.parse_args()
    index, files, boxes = permitted(args.experiment.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler(index, files, boxes))
    print(f"Original MAMMA fitting review: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
