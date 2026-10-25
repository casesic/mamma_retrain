"""Serve only the fixed 40-pair demo-stitch comparison on loopback."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "outputs/experiments/real_riding_stitch_threshold085_20261025"
PAGE = ROOT / "review/real_stitch.html"


def permitted(experiment):
    index = json.loads((experiment / "index.json").read_text())
    if index["schema"] != "real_riding_demo_stitch_comparison_v1":
        raise ValueError("invalid_stitch_index")
    selection = json.loads(Path(index["input_selection"]).read_text())
    targets = json.loads(Path(index["input_targets"]).read_text())
    sources = {(row["dataset"], row["source_frame"], row["camera"]):
               Path(row["source_png"]).resolve() for row in selection["images"]}
    source_files = {}
    output_files = {}
    for number, pair in enumerate(index["pairs"]):
        for camera in ("cam2", "cam4"):
            key = (pair["dataset"], pair["source_frame"], camera)
            source_files[f"/source/{number}/{camera}"] = sources[key]
        for branch, detail in pair["branches"].items():
            for route, field in (("frame", "image"), ("preview", "preview")):
                output = (experiment / detail[field]).resolve()
                if not output.is_relative_to(experiment.resolve()):
                    raise ValueError("stitch_output_outside_experiment")
                output_files[f"/{route}/{number}/{branch}"] = output
    files = {**source_files, **output_files}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("missing_stitch_or_source_image")
    boxes = {row["target_id"]: row["box_xyxy"] for row in targets["targets"]}
    return index, files, boxes


def handler(index, files, boxes):
    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                file, mime = PAGE, "text/html; charset=utf-8"
            elif path == "/data":
                data = json.dumps({"index": index, "target_boxes": boxes},
                                  ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            elif path in files:
                file, mime = files[path], "image/png"
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            if not file.is_file():
                self.send_error(404)
                return
            data = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store" if path == "/" else "private, max-age=3600")
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18872)
    args = parser.parse_args()
    index, files, boxes = permitted(args.experiment.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler(index, files, boxes))
    print(f"Demo stitch review: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
