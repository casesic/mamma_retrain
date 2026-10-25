"""Freeze one pass over every supervised training observation, without reading images."""

import argparse
import json
from pathlib import Path

from mamma_ext.policy import load_active_policy
from mamma_ext.sampler import make_one_epoch_plan
from mamma_ext.split import validate_disjoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, train = load_active_policy(args.config, "train")
    _, validation = load_active_policy(args.config, "validation")
    validate_disjoint(train, validation)
    if config["sampling"]["mode"] != "one_epoch_v1" or Path(args.output).resolve() != Path(config["sampling"]["plan"]):
        raise ValueError("one_epoch_plan_output_must_match_config")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    plan = make_one_epoch_plan(train, config["sampler_seed"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"plan_id": plan["plan_id"], "trainable": len(plan["entries"]),
                      "excluded": len(plan["excluded_no_supervision"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
