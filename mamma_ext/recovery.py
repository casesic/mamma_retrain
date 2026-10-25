"""Full deterministic update-boundary state for bounded MAMMA training."""

import json
import random
from pathlib import Path

import numpy as np
import torch

REGULAR_SCHEMA = "mamma_regular_resume_v7"
ONE_EPOCH_SCHEMA = "mamma_regular_resume_v8"


def _checkpoint_schema(refs):
    return ONE_EPOCH_SCHEMA if refs.get("training_schema") == "mamma_training_v7" else REGULAR_SCHEMA


def capture_rng():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": n[0],
            "keys": n[1].tolist(),
            "pos": n[2],
            "has_gauss": n[3],
            "cached_gaussian": n[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(s):
    random.setstate(s["python"])
    n = s["numpy"]
    np.random.set_state(
        (
            n["name"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["pos"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    torch.set_rng_state(s["torch_cpu"])
    if s["torch_cuda"]:
        if not torch.cuda.is_initialized():
            raise ValueError("checkpoint_cuda_rng_requires_initialized_cuda")
        torch.cuda.set_rng_state_all(s["torch_cuda"])


def regular_payload(model_state, optimizer_state, global_update, rng, sampler, refs):
    """Current regular-run update boundary."""
    required = (
        "config",
        "train_manifests",
        "validation_manifests",
        "model_config",
        "tensor_state",
        "crop_policy_version",
        "gt_policy",
        "augmentation",
        "training_schema",
        "loss_enabled",
        "visibility_reduction",
        "optimizer",
        "sampler_seed",
        "sampling",
        "sampling_plan_id",
    )
    if (
        not isinstance(global_update, int)
        or global_update < 0
        or any(not refs.get(k) for k in required)
        or (refs.get("training_schema") == "mamma_training_v7" and
            (type(refs.get("batch_size")) is not int or refs["batch_size"] < 1))
    ):
        raise ValueError("incomplete_regular_checkpoint_boundary")
    if not isinstance(sampler, dict) or set(sampler) != {"seed", "epoch", "next_position"}:
        raise ValueError("incomplete_regular_sampler_state")
    if any(type(sampler[key]) is not int or sampler[key] < 0 for key in sampler):
        raise ValueError("invalid_regular_sampler_state")
    return {
        "schema": _checkpoint_schema(refs),
        "not_deployable": True,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "global_update": global_update,
        "rng": rng,
        "sampler": sampler,
        "refs": refs,
    }


def save_regular_boundary(path, model, optimizer, global_update, rng, sampler, refs):
    payload = regular_payload(
        model.state_dict(), optimizer.state_dict(), global_update, rng, sampler, refs
    )
    torch.save(payload, path)


def load_regular_boundary(path, model, optimizer, expected_refs):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != _checkpoint_schema(expected_refs)
        or payload.get("not_deployable") is not True
        or payload.get("refs") != expected_refs
    ):
        raise ValueError("regular_resume_schema_or_refs_mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng(payload["rng"])
    return {
        "global_update": payload["global_update"],
        "sampler": payload["sampler"],
        "refs": payload["refs"],
    }


def load_evaluation_boundary(path, model, config, validation_manifests):
    """Load model weights only after checking the current evaluation contract."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != _checkpoint_schema(payload.get("refs", {})) or payload.get("not_deployable") is not True:
        raise ValueError("unsupported_evaluation_checkpoint")
    refs = payload.get("refs", {})
    if (
        refs.get("model_config") != str(Path(config["model_config"]).resolve())
        or refs.get("tensor_state") != str(Path(config["tensor_state"]).resolve())
        or refs.get("crop_policy_version") != config["crop_policy_version"]
        or refs.get("gt_policy") != "saved_mamma_512_npz_v1"
        or refs.get("augmentation") != config["augmentation"]
        or refs.get("training_schema") != config["schema"]
        or refs.get("loss_enabled") != config["loss_enabled"]
        or refs.get("visibility_reduction") != config["visibility_reduction"]
        or refs.get("optimizer") != config["optimizer"]
        or refs.get("sampler_seed") != config["sampler_seed"]
        or refs.get("sampling") != config["sampling"]
        or (refs.get("batch_size") != config.get("batch_size")
            if config["schema"] == "mamma_training_v7" else "batch_size" in refs)
        or not refs.get("sampling_plan_id")
        or refs.get("validation_manifests") != validation_manifests
    ):
        raise ValueError("evaluation_checkpoint_contract_mismatch")
    if config["sampling"]["mode"] == "occlusion_quota_v1":
        plan_id = json.loads(Path(config["sampling"]["plan"]).read_text())["plan_id"]
        if refs["sampling_plan_id"] != plan_id:
            raise ValueError("evaluation_sampling_plan_mismatch")
    elif config["sampling"]["mode"] == "one_epoch_v1":
        plan_id = json.loads(Path(config["sampling"]["plan"]).read_text())["plan_id"]
        if refs["sampling_plan_id"] != plan_id:
            raise ValueError("evaluation_sampling_plan_mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    return refs


def load_frozen_20261024_point_mean_state(path, model, config, validation_manifests):
    """Read only the known first 500-step model, without treating it as a resumable v6 run."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    refs = payload.get("refs", {})
    if (
        payload.get("schema") != "mamma_regular_resume_v5"
        or payload.get("not_deployable") is not True
        or payload.get("global_update") != 500
        or refs.get("training_schema") != "mamma_training_v4"
        or "visibility_reduction" in refs
        or refs.get("model_config") != str(Path(config["model_config"]).resolve())
        or refs.get("tensor_state") != str(Path(config["tensor_state"]).resolve())
        or refs.get("validation_manifests") != validation_manifests
        or refs.get("crop_policy_version") != config["crop_policy_version"]
        or refs.get("gt_policy") != "saved_mamma_512_npz_v1"
        or refs.get("augmentation") != config["augmentation"]
        or refs.get("loss_enabled") != config["loss_enabled"]
        or refs.get("optimizer") != config["optimizer"]
        or refs.get("sampler_seed") != config["sampler_seed"]
        or config["visibility_reduction"] != "point_mean"
    ):
        raise ValueError("frozen_point_mean_model_contract_mismatch")
    model.load_state_dict(payload["model_state"], strict=True)


def load_frozen_20261024_class_mean_state(path, model, config, validation_manifests):
    """Read the completed historical class-mean weights, never its optimizer or RNG."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    refs = payload.get("refs", {})
    if (
        payload.get("schema") != "mamma_regular_resume_v6"
        or payload.get("not_deployable") is not True
        or payload.get("global_update") != 500
        or refs.get("training_schema") != "mamma_training_v5"
        or refs.get("visibility_reduction") != "class_mean"
        or refs.get("model_config") != str(Path(config["model_config"]).resolve())
        or refs.get("tensor_state") != str(Path(config["tensor_state"]).resolve())
        or refs.get("validation_manifests") != validation_manifests
        or refs.get("crop_policy_version") != config["crop_policy_version"]
        or refs.get("gt_policy") != "saved_mamma_512_npz_v1"
        or refs.get("augmentation") != config["augmentation"]
        or refs.get("loss_enabled") != config["loss_enabled"]
        or refs.get("optimizer") != config["optimizer"]
        or refs.get("sampler_seed") != config["sampler_seed"]
        or config["visibility_reduction"] != "class_mean"
    ):
        raise ValueError("frozen_class_mean_model_contract_mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
