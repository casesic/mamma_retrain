"""Full deterministic update-boundary state for bounded MAMMA training."""

import random
from pathlib import Path

import numpy as np
import torch

REGULAR_SCHEMA = "mamma_regular_resume_v3"


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
        "train_manifest",
        "validation_manifest",
        "model_config",
        "tensor_state",
        "crop_policy_version",
        "gt_policy",
        "augmentation",
        "training_schema",
        "loss_enabled",
    )
    if (
        not isinstance(global_update, int)
        or global_update < 0
        or any(not refs.get(k) for k in required)
    ):
        raise ValueError("incomplete_regular_checkpoint_boundary")
    if not isinstance(sampler, dict) or not {"epoch", "next_position"} <= sampler.keys():
        raise ValueError("incomplete_regular_sampler_state")
    return {
        "schema": REGULAR_SCHEMA,
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
        payload.get("schema") != REGULAR_SCHEMA
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


def load_evaluation_boundary(path, model, config):
    """Load model weights only after checking the current evaluation contract."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != REGULAR_SCHEMA or payload.get("not_deployable") is not True:
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
    ):
        raise ValueError("evaluation_checkpoint_contract_mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
