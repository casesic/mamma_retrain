"""Loopback-only, read-only server for the fixed real-riding human review."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
PAGE = ROOT / "review/real_riding.html"
BALANCED = ROOT / "outputs/experiments/city_full_visibility_class_mean_20261024/review_predictions_class_mean_500.json"
OCCLUSION30 = ROOT / "outputs/experiments/city_full_occlusion_30pct_20261025/review_predictions_class_mean_occlusion30_500.json"
MERGED30 = ROOT / "outputs/experiments/city_full_plus_occlusion_30pct_20261025/review_predictions_class_mean_merged_occlusion30_500.json"
FULL_EPOCH = ROOT / "outputs/experiments/city_merged_one_epoch_20261025/review_predictions_class_mean_merged_full_epoch.json"
SELF_OCCLUSION = ROOT / "outputs/experiments/self_occlusion_unknown_trial_20261025/review_predictions_self_occlusion_full_epoch.json"


def permitted_files():
    selection = json.loads((EXPERIMENT / "selection.json").read_text())
    targets = json.loads((EXPERIMENT / "targets.json").read_text())
    images = {row["image_id"]: Path(row["source_png"]).resolve() for row in selection["images"]}
    masks = {row["target_id"]: (EXPERIMENT / row["mask"]).resolve() for row in targets["targets"]}
    for file in (*images.values(), *masks.values()):
        if not file.is_file():
            raise FileNotFoundError(file)
    return images, masks


def handler(images, masks):
    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                file, mime = PAGE, "text/html; charset=utf-8"
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            elif path == "/data":
                file, mime = EXPERIMENT / "review_data.json", "application/json"
            elif path == "/reveal":
                file, mime = EXPERIMENT / "mapping.json", "application/json"
            elif path == "/balanced":
                file, mime = BALANCED, "application/json"
            elif path == "/occlusion30":
                file, mime = OCCLUSION30, "application/json"
            elif path == "/merged30":
                file, mime = MERGED30, "application/json"
            elif path == "/full-epoch":
                file, mime = FULL_EPOCH, "application/json"
            elif path == "/self-occlusion":
                file, mime = SELF_OCCLUSION, "application/json"
            elif path.startswith("/image/") and path[len("/image/"):] in images:
                file, mime = images[path[len("/image/"):]], "image/png"
            elif path.startswith("/mask/") and path[len("/mask/"):] in masks:
                file, mime = masks[path[len("/mask/"):]], "image/png"
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
            self.send_header("Cache-Control", "no-store" if path in ("/", "/data", "/reveal", "/balanced", "/occlusion30", "/merged30", "/full-epoch", "/self-occlusion") else "private, max-age=3600")
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=18868)
    args = parser.parse_args()
    images, masks = permitted_files()
    server = ThreadingHTTPServer((args.host, args.port), handler(images, masks))
    print(f"Review page: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
