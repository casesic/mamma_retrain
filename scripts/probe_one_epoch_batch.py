"""Three disposable real optimizer steps to measure the chosen true batch size."""

import argparse
import json
import time
from pathlib import Path

import torch

from mamma_ext.policy import active_task_loss, configure_active_optimizer, load_active_policy
from mamma_ext.runner import _configure_numeric, _load_original_model, _tensor_batch
from mamma_ext.sampler import load_one_epoch_plan
from mamma_ext.tasks import set_training_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size not in (4, 2, 1) or Path(args.output).exists():
        raise ValueError("unsupported_or_existing_probe")
    config, dataset = load_active_policy(args.config, "train")
    plan = load_one_epoch_plan(config["sampling"]["plan"], dataset, config["sampler_seed"])
    entries = plan["entries"]
    occluded = next(i for i, row in enumerate(entries) if row["negative"])
    coordinate_only = next(i for i, row in enumerate(entries)
                           if row["coordinates"] and not row["positive"] and not row["negative"])
    starts = [0, occluded, coordinate_only]
    _configure_numeric()
    device = torch.device(args.device)
    model, strict = _load_original_model(config, device)
    optimizer, _ = configure_active_optimizer(model, config)
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    started = time.monotonic()
    for start in starts:
        selected = entries[start:start + args.batch_size]
        batch = _tensor_batch([dataset[row["dataset_index"]] for row in selected], device)
        set_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        result = active_task_loss(model(batch["img"], batch["mask"]), batch, config)
        if result["skip_optimizer"] or not torch.isfinite(result["loss"]):
            raise ValueError("invalid_memory_probe_batch")
        result["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        records.append({"observation_ids": [row["observation_id"] for row in selected],
                        "occluded": sum(row["negative"] > 0 for row in selected),
                        "coordinate_only": sum(not row["positive"] and not row["negative"] for row in selected),
                        "loss": float(result["loss"].detach().cpu())})
        print(f"probe {len(records)}/3 complete", flush=True)
    report = {"schema": "mamma_batch_memory_probe_v1", "device": str(device),
              "device_name": torch.cuda.get_device_name(device), "batch_size": args.batch_size,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "elapsed_seconds": time.monotonic() - started, "strict_load": strict,
              "disposable_updates": records}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("batch_size", "peak_allocated_bytes",
          "peak_reserved_bytes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
