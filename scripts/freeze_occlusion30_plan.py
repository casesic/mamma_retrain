"""Write the fixed 150/350 training schedule once, using only train GT metadata."""

import argparse
import json
from pathlib import Path

from mamma_ext.policy import load_active_policy
from mamma_ext.sampler import make_occlusion_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/retrain/mamma_city_full_occlusion_30pct_20261025.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    args = parser.parse_args()
    config, dataset = load_active_policy(args.config, split="train")
    sampling = config["sampling"]
    if sampling["mode"] != "occlusion_quota_v1":
        raise ValueError("occlusion_quota_config_required")
    path = Path(sampling["plan"])
    if path.exists():
        raise FileExistsError(path)
    plan = make_occlusion_plan(dataset, config["sampler_seed"],
                               sampling["total_updates"], sampling["occlusion_updates"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(plan, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"path": str(path), "plan_id": plan["plan_id"],
                      "pool_sizes": plan["pool_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
