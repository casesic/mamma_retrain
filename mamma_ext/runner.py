"""Regular MAMMA plan, bounded train and ERP evaluation orchestration.

Importing this module never builds a model or reads a checkpoint. Current
configuration rejects execution; later runs require a separate authorized
configuration and explicit CLI arguments.
"""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .policy import load_active_policy
from .sampler import mixed_order
from .split import group_inventory, validate_disjoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/retrain/mamma_regular.json"


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def read_plan(config_path, split="train"):
    cfg, dataset = load_active_policy(config_path, split=split)
    return (
        cfg,
        dataset,
        {
            "schema": "mamma_regular_plan_v1",
            "configuration": str(Path(config_path).resolve()),
            "input_config": cfg["input_config"],
            "split": split,
            "manifests": dataset.cfg["manifests"],
            "crop_policy_version": cfg["crop_policy_version"],
            "gt_policy": "saved_mamma_512_npz_v1",
            "target_grouping": group_inventory(dataset),
            "loss_enabled": cfg["loss_enabled"],
            "visibility_reduction": cfg["visibility_reduction"],
            "sampling": cfg["sampling"],
            "batch_size": cfg.get("batch_size", 1),
            "model_heads": cfg["model_heads"],
            "visibility_label_qualification": cfg["label_qualification"],
            "execution_authorized": cfg["execution_authorized"],
            "model_loaded": False,
            "image_reads": 0,
        },
    )


def _execution_preflight(args, cfg, mode):
    if not args.execute:
        raise ValueError("explicit_execute_required")
    if cfg["execution_authorized"] is not True:
        raise ValueError("configuration_execution_not_authorized")
    if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
        raise ValueError("dotenv_must_be_disabled_for_execution")
    if not args.output or not args.device:
        raise ValueError("explicit_output_and_device_required")
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output.exists():
        raise ValueError("output_must_be_new_directory_under_workspace_outputs")
    if not cfg.get("model_config") or not cfg.get("tensor_state"):
        raise ValueError("explicit_model_config_and_tensor_state_required")
    if mode == "train":
        if (
            not args.max_updates
            or args.max_updates <= 0
            or not args.checkpoint_every
            or args.checkpoint_every <= 0
        ):
            raise ValueError("positive_update_budget_and_checkpoint_interval_required")
        if cfg["loss_enabled"]["visibility"] and cfg.get("label_qualification") != "qualified":
            raise ValueError("visibility_training_labels_not_qualified")
    return output


def _tensor_batch(items, device):
    import torch

    keys = (
        "img",
        "mask",
        "xy_target",
        "coordinate_loss_valid",
        "visibility",
        "visibility_loss_valid",
    )
    if isinstance(items, dict):
        items = [items]
    batch = {key: torch.as_tensor(np.stack([item[key] for item in items]), device=device)
             for key in keys}
    for key in ("img", "mask", "xy_target"):
        batch[key] = batch[key].float()
    return batch


def _configure_numeric():
    # CUDA matmul requires this before the first cuBLAS call in deterministic mode.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _load_original_model(cfg, device):
    from .model import deployment_config, strict_model_from_tensors, model_config_summary

    model_cfg = deployment_config(cfg["model_config"])
    summary = model_config_summary(model_cfg)
    if (
        summary["num_joints"] != 512
        or summary["image_size"] != [384, 512]
        or not summary["normalize_plus_min_one"]
    ):
        raise ValueError("incompatible_mammanet_model_configuration")
    model, report = strict_model_from_tensors(model_cfg, cfg["tensor_state"])
    if model is None or not report["strict_load"]:
        raise ValueError("strict_original_state_load_failed:" + repr(report))
    return model.to(device), report


def _refs(args, cfg, train_manifests, validation_manifests, sampling_plan_id):
    return {
        "config": str(Path(args.config).resolve()),
        "train_manifests": train_manifests,
        "validation_manifests": validation_manifests,
        "model_config": str(Path(cfg["model_config"]).resolve()),
        "tensor_state": str(Path(cfg["tensor_state"]).resolve()),
        "crop_policy_version": cfg["crop_policy_version"],
        "gt_policy": "saved_mamma_512_npz_v1",
        "augmentation": cfg["augmentation"],
        "training_schema": cfg["schema"],
        "loss_enabled": cfg["loss_enabled"],
        "visibility_reduction": cfg["visibility_reduction"],
        "optimizer": cfg["optimizer"],
        "sampler_seed": cfg["sampler_seed"],
        "sampling": cfg["sampling"],
        "sampling_plan_id": sampling_plan_id,
        **({"batch_size": cfg["batch_size"]} if "batch_size" in cfg else {}),
    }


def visibility_metrics(labels, logits):
    """Do not turn a one-class observation into an apparent perfect score."""
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if labels.shape != logits.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid_visibility_metric_inputs")
    positive = int(np.count_nonzero(labels == 1))
    negative = int(np.count_nonzero(labels == 0))
    result = {
        "positive": positive,
        "negative": negative,
        "known": len(labels),
        "nonfinite_logit_count": int(np.count_nonzero(~np.isfinite(logits))),
        "status": (
            "two_class_metric_available"
            if positive and negative
            else "single_class_or_unlabeled_no_accuracy"
        ),
    }
    if positive and negative:
        if not np.isfinite(logits).all():
            result["status"] = "nonfinite_logits_no_accuracy"
            return result
        result["accuracy"] = float(np.mean((logits > 0) == labels))
        result["positive_recall"] = float(np.mean(logits[labels == 1] > 0))
        result["negative_recall"] = float(np.mean(logits[labels == 0] <= 0))
    return result


def error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "points": int(len(values)),
        "mean_erp_px": (float("inf") if not np.isfinite(values).all()
                        else float(values.mean()) if len(values) else None),
    }


def train(args, cfg, dataset):
    if cfg["sampling"]["mode"] == "one_epoch_v1":
        return train_one_epoch(args, cfg, dataset)
    output = _execution_preflight(args, cfg, "train")
    if not len(dataset):
        raise ValueError("no_observed_training_targets")
    _, validation = load_active_policy(args.config, split="validation")
    groups = validate_disjoint(dataset, validation)
    fixed_plan = None
    if cfg["sampling"]["mode"] == "occlusion_quota_v1":
        from .sampler import load_occlusion_plan
        fixed_plan = load_occlusion_plan(cfg["sampling"]["plan"], dataset, cfg["sampler_seed"])
        if args.max_updates > len(fixed_plan["entries"]):
            raise ValueError("update_budget_exceeds_fixed_sampling_plan")
    import torch
    from .policy import active_task_loss, configure_active_optimizer
    from .tasks import set_training_mode
    from .recovery import capture_rng, load_regular_boundary, save_regular_boundary

    _configure_numeric()
    device = torch.device(args.device)
    model, strict_report = _load_original_model(cfg, device)
    optimizer, parameter_groups = configure_active_optimizer(model, cfg)
    frozen_head_start = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(("decoder.contact_prob.", "decoder.floor_contact_prob."))
    }
    refs = _refs(args, cfg, dataset.cfg["manifests"], validation.cfg["manifests"],
                 fixed_plan["plan_id"] if fixed_plan else "episode_round_robin")
    update, epoch, position = 0, 0, 0
    sampler_seed = cfg["sampler_seed"]
    if args.checkpoint:
        boundary = load_regular_boundary(args.checkpoint, model, optimizer, refs)
        update = boundary["global_update"]
        epoch = boundary["sampler"]["epoch"]
        position = boundary["sampler"]["next_position"]
        if boundary["sampler"]["seed"] != sampler_seed:
            raise ValueError("sampler_seed_mismatch")
    order_length = len(fixed_plan["entries"]) if fixed_plan else len(dataset)
    if position < 0 or position >= order_length or (fixed_plan and epoch != 0):
        raise ValueError("sampler_position_out_of_range")
    if fixed_plan and position != update:
        raise ValueError("fixed_sampler_position_update_mismatch")
    if update >= args.max_updates:
        raise ValueError("resume_already_at_or_beyond_update_budget")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "run_plan.json",
        {
            "schema": "mamma_regular_run_v1",
            "status": "explicit_execution",
            "max_updates": args.max_updates,
            "checkpoint_every": args.checkpoint_every,
            "strict_load": strict_report,
            "parameter_groups": parameter_groups,
            "groups": groups,
            "refs": refs,
            "sampler_seed": sampler_seed,
            "sampling": cfg["sampling"],
            "sampling_plan_id": refs["sampling_plan_id"],
        },
    )
    skipped = 0
    started = time.monotonic()
    order = (None if fixed_plan else mixed_order(dataset, sampler_seed, epoch))
    with (output / "updates.jsonl").open("w") as log:
        while update < args.max_updates:
            plan_entry = fixed_plan["entries"][position] if fixed_plan else None
            item = dataset[plan_entry["dataset_index"] if plan_entry else order[position]]
            batch = _tensor_batch(item, device)
            set_training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["img"], batch["mask"])
            result = active_task_loss(prediction, batch, cfg)
            next_position = position + 1
            next_epoch = epoch
            if next_position == order_length:
                next_position = 0
                next_epoch += 1
            if result["skip_optimizer"]:
                if fixed_plan:
                    raise ValueError("fixed_sampling_plan_contains_skipped_update:" + item["identity"]["observation_id"])
                skipped += 1
                if skipped >= len(dataset):
                    raise RuntimeError("entire_epoch_without_valid_update")
            else:
                skipped = 0
                if not torch.isfinite(result["loss"]):
                    raise ValueError("nonfinite_training_loss")
                result["loss"].backward()
                inspect = update < 3 or (update + 1) % args.checkpoint_every == 0
                diagnostic = {}
                if inspect:
                    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in named_trainable):
                        raise ValueError("nonfinite_training_gradient")
                    first_name, first_parameter = named_trainable[0]
                    before = first_parameter.detach().clone()
                    diagnostic["tracked_parameter"] = first_name
                    diagnostic["tracked_gradient_max_abs"] = (
                        float(first_parameter.grad.detach().abs().max())
                        if first_parameter.grad is not None else None
                    )
                optimizer.step()
                update += 1
                if inspect:
                    diagnostic["tracked_update_max_abs"] = float(
                        (first_parameter.detach() - before).abs().max()
                    )
            event = {
                "global_update": update,
                "epoch": epoch,
                "position": position,
                "observation_id": item["identity"]["observation_id"],
                "sampling_pool": plan_entry["pool"] if plan_entry else "episode_round_robin",
                "episode": item["identity"]["episode"],
                "loss": float(result["loss"].detach().cpu()),
                "coordinate_loss": float(result["parts"]["coordinate"].detach().cpu()),
                "visibility_loss": float(result["parts"]["visibility"].detach().cpu()),
                "visibility_reduction": cfg["visibility_reduction"],
                "counts": result["counts"],
                "skip_optimizer": result["skip_optimizer"],
                "skip_reason": result["skip_reason"],
            }
            if not result["skip_optimizer"]:
                event.update(diagnostic)
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            position, epoch = next_position, next_epoch
            if next_epoch != event["epoch"] and not fixed_plan:
                order = mixed_order(dataset, sampler_seed, epoch)
            if (
                update
                and not result["skip_optimizer"]
                and (update % args.checkpoint_every == 0 or update == args.max_updates)
            ):
                save_regular_boundary(
                    output / f"checkpoint_update_{update:06d}.pt",
                    model,
                    optimizer,
                    update,
                    capture_rng(),
                    {"seed": sampler_seed, "epoch": epoch, "next_position": position},
                    refs,
                )
    for name, parameter in model.named_parameters():
        if name in frozen_head_start and not torch.equal(parameter.detach().cpu(), frozen_head_start[name]):
            raise AssertionError("contact_head_changed:" + name)
    write_json(
        output / "completed.json",
        {"global_update": update, "status": "complete", "not_deployable": True,
         "groups": groups, "elapsed_seconds": time.monotonic() - started,
         "frozen_contact_heads_unchanged": True},
    )


def train_one_epoch(args, cfg, dataset):
    """Consume a frozen one-pass sample order in real, possibly partial batches."""
    output = _execution_preflight(args, cfg, "train")
    from .sampler import load_one_epoch_plan
    from .policy import active_task_loss, configure_active_optimizer
    from .tasks import set_training_mode
    from .recovery import capture_rng, load_regular_boundary, save_regular_boundary
    import torch

    _, validation = load_active_policy(args.config, split="validation")
    groups = validate_disjoint(dataset, validation)
    plan = load_one_epoch_plan(cfg["sampling"]["plan"], dataset, cfg["sampler_seed"])
    entries = plan["entries"]
    batch_size = cfg["batch_size"]
    total_updates = (len(entries) + batch_size - 1) // batch_size
    if not entries or args.max_updates > total_updates:
        raise ValueError("update_budget_exceeds_one_epoch")
    _configure_numeric()
    device = torch.device(args.device)
    model, strict_report = _load_original_model(cfg, device)
    optimizer, parameter_groups = configure_active_optimizer(model, cfg)
    refs = _refs(args, cfg, dataset.cfg["manifests"], validation.cfg["manifests"], plan["plan_id"])
    update, position = 0, 0
    if args.checkpoint:
        boundary = load_regular_boundary(args.checkpoint, model, optimizer, refs)
        update = boundary["global_update"]
        sampler = boundary["sampler"]
        position = sampler["next_position"]
        if sampler["seed"] != cfg["sampler_seed"] or sampler["epoch"] != 0:
            raise ValueError("one_epoch_resume_sampler_mismatch")
    if position != min(update * batch_size, len(entries)) or update >= args.max_updates:
        raise ValueError("one_epoch_resume_position_or_budget_mismatch")
    frozen_head_start = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                         if name.startswith(("decoder.contact_prob.", "decoder.floor_contact_prob."))}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_plan.json", {
        "schema": "mamma_one_epoch_run_v1", "max_updates": args.max_updates,
        "total_updates": total_updates, "trainable_observations": len(entries),
        "excluded_no_supervision": len(plan["excluded_no_supervision"]),
        "batch_size": batch_size, "start_update": update, "start_sample_position": position,
        "checkpoint_every": args.checkpoint_every, "strict_load": strict_report,
        "parameter_groups": parameter_groups, "groups": groups, "refs": refs,
    })
    started = time.monotonic()
    with (output / "updates.jsonl").open("w") as log:
        while update < args.max_updates:
            selected = entries[position:position + batch_size]
            if not selected:
                raise ValueError("one_epoch_exhausted_before_update_budget")
            items = [dataset[entry["dataset_index"]] for entry in selected]
            for item, entry in zip(items, selected):
                if item["identity"]["observation_id"] != entry["observation_id"]:
                    raise ValueError("one_epoch_observation_identity_changed")
            batch = _tensor_batch(items, device)
            set_training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["img"], batch["mask"])
            result = active_task_loss(prediction, batch, cfg)
            if result["skip_optimizer"] or not torch.isfinite(result["loss"]):
                raise ValueError("one_epoch_frozen_batch_has_no_finite_supervision")
            result["loss"].backward()
            if update < 3 or (update + 1) % args.checkpoint_every == 0:
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad):
                    raise ValueError("nonfinite_training_gradient")
            optimizer.step()
            position += len(selected)
            update += 1
            log.write(json.dumps({
                "global_update": update, "sample_position_after": position,
                "batch_size": len(selected), "observation_ids": [e["observation_id"] for e in selected],
                "episodes": [e["episode"] for e in selected],
                "occluded_observations": sum(e["negative"] > 0 for e in selected),
                "loss": float(result["loss"].detach().cpu()),
                "coordinate_loss": float(result["parts"]["coordinate"].detach().cpu()),
                "visibility_loss": float(result["parts"]["visibility"].detach().cpu()),
                "counts": result["counts"],
            }, ensure_ascii=False) + "\n")
            log.flush()
            if update % args.checkpoint_every == 0 or update == args.max_updates:
                save_regular_boundary(output / f"checkpoint_update_{update:06d}.pt",
                                      model, optimizer, update, capture_rng(),
                                      {"seed": cfg["sampler_seed"], "epoch": 0,
                                       "next_position": position}, refs)
    for name, parameter in model.named_parameters():
        if name in frozen_head_start and not torch.equal(parameter.detach().cpu(), frozen_head_start[name]):
            raise AssertionError("contact_head_changed:" + name)
    write_json(output / "completed.json", {
        "status": "complete_epoch" if position == len(entries) else "checkpointed_partial_epoch",
        "global_update": update, "samples_consumed": position,
        "trainable_observations": len(entries), "elapsed_seconds": time.monotonic() - started,
        "frozen_contact_heads_unchanged": True, "not_deployable": True,
    })


def evaluate(args, cfg, dataset):
    output = _execution_preflight(args, cfg, "evaluate")
    if not len(dataset):
        raise ValueError("no_observed_evaluation_targets")
    import torch
    from .model import predict_one, restore_xy, summarize_errors
    from .recovery import load_evaluation_boundary

    _configure_numeric()
    device = torch.device(args.device)
    model, strict_report = _load_original_model(cfg, device)
    checkpoint_refs = None
    # A checkpoint is tied to its training split, while read-only evaluation
    # may use another episode-disjoint validation set.
    _, training_validation = load_active_policy(args.config, split="validation")
    if args.checkpoint:
        checkpoint_refs = load_evaluation_boundary(
            args.checkpoint, model, cfg, training_validation.cfg["manifests"]
        )
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    all_errors = []
    errors_by_visibility = {name: [] for name in ("visible", "occluded", "unknown")}
    errors_by_episode = {}
    vis_labels = []
    vis_logits = []
    predictions = []
    targets = []
    valid_masks = []
    visibility_values = []
    visibility_outputs = []
    for index in range(len(dataset)):
        item = dataset[index]
        prediction, _ = predict_one(model, item, device)
        raw = prediction["joints2d"][0]
        model_xy, erp, affine_erp = restore_xy(raw[:, :2], item, True)
        restore_delta = erp - affine_erp
        if item.get("periodic_x", False):
            width = int(item["full_size_wh"][0])
            restore_delta[:, 0] = (restore_delta[:, 0] + width / 2) % width - width / 2
        if np.max(np.abs(restore_delta)) > 0.01:
            raise ValueError("deployment_affine_restoration_disagree")
        errors, metrics = summarize_errors(
            erp, item["xy_erp"], item["coordinate_loss_valid"], int(item["full_size_wh"][0])
        )
        all_errors.extend(errors[item["coordinate_loss_valid"]].tolist())
        episode = item["identity"]["episode"]
        errors_by_episode.setdefault(episode, []).extend(errors[item["coordinate_loss_valid"]].tolist())
        for name, value in (("visible", 1), ("occluded", 0), ("unknown", -1)):
            selected = item["coordinate_loss_valid"] & (item["visibility"] == value)
            errors_by_visibility[name].extend(errors[selected].tolist())
        valid = item["visibility_loss_valid"]
        if valid.any():
            vis_labels.extend(item["visibility"][valid].tolist())
            vis_logits.extend(prediction["visibility"][0, :, 0][valid].tolist())
        predictions.append(erp.astype(np.float32))
        targets.append(item["xy_erp"].astype(np.float32))
        valid_masks.append(item["coordinate_loss_valid"])
        visibility_values.append(item["visibility"])
        visibility_outputs.append(prediction["visibility"][0, :, 0].astype(np.float32))
        rows.append(
            {
                "identity": item["identity"],
                "metrics": metrics,
                "prediction_index": index,
            }
        )
    np.savez_compressed(
        output / "predictions.npz",
        vertex_ids=dataset.vertex_ids,
        predicted_erp_xy=np.stack(predictions),
        gt_erp_xy=np.stack(targets),
        coordinate_loss_valid=np.stack(valid_masks),
        visibility=np.stack(visibility_values),
        visibility_logits=np.stack(visibility_outputs),
    )
    visibility = visibility_metrics(vis_labels, vis_logits)
    errors = np.asarray(all_errors)
    sample_means = [
        r["metrics"]["mean_erp_px"] for r in rows if r["metrics"]["mean_erp_px"] is not None
    ]
    write_json(
        output / "metrics.json",
        {
            "schema": "mamma_regular_eval_v1",
            "rows": rows,
            "point_weighted_mean_erp_px": (
                float("inf")
                if not np.isfinite(errors).all()
                else float(np.mean(errors)) if len(errors) else None
            ),
            "sample_equal_mean_erp_px": float(np.mean(sample_means)) if sample_means else None,
            "sample_mean_denominator": len(sample_means),
            "visibility": visibility,
            "coordinate_by_visibility": {key: error_summary(value) for key, value in errors_by_visibility.items()},
            "coordinate_by_episode": {key: error_summary(value) for key, value in errors_by_episode.items()},
            "strict_load": strict_report,
            "crop_policy_version": cfg["crop_policy_version"],
            "gt_policy": "saved_mamma_512_npz_v1",
            "evaluation_manifests": dataset.cfg["manifests"],
            "model_training_validation_manifests": training_validation.cfg["manifests"],
            "model_training_manifests": checkpoint_refs["train_manifests"] if checkpoint_refs else None,
            "model_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "check-data", "train", "evaluate"):
        part = sub.add_parser(name)
        part.add_argument("--config", default=str(DEFAULT_CONFIG))
        if name in ("plan", "check-data", "evaluate"):
            part.add_argument("--split", choices=("train", "validation"), default="train")
        if name == "check-data":
            part.add_argument("--limit", type=int, default=2)
        if name in ("train", "evaluate"):
            part.add_argument("--output", required=True)
            part.add_argument("--device", required=True)
            part.add_argument("--checkpoint")
            part.add_argument("--execute", action="store_true")
        if name == "train":
            part.add_argument("--max-updates", type=int, required=True)
            part.add_argument("--checkpoint-every", type=int, required=True)
        if name == "evaluate":
            part.add_argument("--eval-manifest", action="append")
    args = parser.parse_args(argv)
    if args.command in ("train", "evaluate"):
        raw_cfg = json.loads(Path(args.config).read_text())
        _execution_preflight(args, raw_cfg, args.command)
    cfg, dataset, plan = read_plan(args.config, getattr(args, "split", "train"))
    if args.command == "evaluate" and args.eval_manifest:
        if args.split != "validation":
            raise ValueError("external_evaluation_requires_validation_split")
        from .data import MammaSavedGTDataset

        train_dataset = load_active_policy(args.config, split="train")[1]
        dataset = MammaSavedGTDataset([str(Path(p).resolve()) for p in args.eval_manifest])
        validate_disjoint(train_dataset, dataset)
    if args.command == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if args.command == "check-data":
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError("bounded_check_limit_required")
        rows = []
        for index in range(args.limit):
            item = dataset[index]
            rows.append(
                {
                    "identity": item["identity"],
                    "gt_source": item["gt_source"],
                    "image_shape": list(item["img"].shape),
                    "mask_shape": list(item["mask"].shape),
                    "coordinate_valid": int(item["coordinate_loss_valid"].sum()),
                    "visibility_valid": int(item["visibility_loss_valid"].sum()),
                    "source_sampling_supported": item["source_sampling_supported"],
                }
            )
        result = {"plan": plan, "checked_targets": rows, "model_loaded": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.command == "train":
        return train(args, cfg, dataset)
    return evaluate(args, cfg, dataset)
