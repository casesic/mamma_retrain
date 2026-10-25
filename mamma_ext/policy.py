"""Validate the one current training policy and bind it to data and tasks."""

import json
from pathlib import Path

from .data import MammaSavedGTDataset
from .tasks import configure_training_parameters, masked_task_loss

TRAINING_SCHEMA = "mamma_training_v7"
POLICY_KEYS = {
    "model_heads",
    "coordinate_scale",
    "residual_cap",
    "optimizer",
    "schema",
    "input_config",
    "crop_policy_version",
    "augmentation",
    "loss_enabled",
    "visibility_reduction",
    "execution_authorized",
    "label_qualification",
    "model_config",
    "tensor_state",
    "sampler_seed",
    "sampling",
}


def _config_path(config_file, reference):
    """Project-owned references are relative to their declaring config file."""
    path = Path(reference)
    return (
        str((config_file.parent / path).resolve())
        if not path.is_absolute()
        else str(path.resolve())
    )


def load_active_policy(path, split="train"):
    config_file = Path(path).resolve()
    config = json.loads(config_file.read_text())
    one_epoch = config.get("schema") == TRAINING_SCHEMA
    if (
        set(config) != (POLICY_KEYS | {"batch_size"} if one_epoch else POLICY_KEYS)
        or config.get("schema") not in (TRAINING_SCHEMA, "mamma_training_v6")
        or config.get("crop_policy_version") != "mask_bbox_periodic_v2"
        or config.get("augmentation") != "off"
        or config.get("visibility_reduction") not in ("point_mean", "class_mean")
        or type(config.get("execution_authorized")) is not bool
        or config.get("model_heads") != ["joints2d", "visibility", "contact", "floor_contact"]
        or type(config.get("sampler_seed")) is not int
        or config["sampler_seed"] < 0
    ):
        raise ValueError("unsupported_active_training_policy")
    if one_epoch and (type(config["batch_size"]) is not int or config["batch_size"] < 1):
        raise ValueError("invalid_training_batch_size")
    losses = config.get("loss_enabled")
    if (
        not isinstance(losses, dict)
        or set(losses) != {"coordinate", "visibility"}
        or losses["coordinate"] is not True
        or type(losses["visibility"]) is not bool
    ):
        raise ValueError("task_loss_optimizer_mismatch")
    for name in ("input_config", "model_config", "tensor_state"):
        config[name] = _config_path(config_file, config[name])
    sampling = config["sampling"]
    if not one_epoch and sampling == {"mode": "episode_round_robin"}:
        pass
    elif (one_epoch and isinstance(sampling, dict)
          and set(sampling) == {"mode", "plan"}
          and sampling["mode"] == "one_epoch_v1"
          and isinstance(sampling["plan"], str) and sampling["plan"]):
        sampling["plan"] = _config_path(config_file, sampling["plan"])
    elif (not one_epoch and isinstance(sampling, dict)
          and set(sampling) == {"mode", "total_updates", "occlusion_updates", "plan"}
          and sampling["mode"] == "occlusion_quota_v1"
          and type(sampling["total_updates"]) is int and sampling["total_updates"] == 500
          and type(sampling["occlusion_updates"]) is int and sampling["occlusion_updates"] == 150
          and isinstance(sampling["plan"], str) and sampling["plan"]):
        sampling["plan"] = _config_path(config_file, sampling["plan"])
    else:
        raise ValueError("unsupported_sampling_policy")
    data_file = Path(config["input_config"])
    source = json.loads(data_file.read_text())
    if (
        set(source) != {"schema", "train", "validation"}
        or source["schema"] != "mamma_data_split_v1"
        or split not in ("train", "validation")
        or not all(isinstance(source[side], list) and source[side]
                   and all(isinstance(p, str) and p for p in source[side])
                   for side in ("train", "validation"))
    ):
        raise ValueError("current_data_split_required")
    manifests = [_config_path(data_file, name) for name in source[split]]
    dataset = MammaSavedGTDataset(manifests)
    return config, dataset


def active_task_loss(prediction, batch, config):
    """Unknown visibility labels are skipped by the task loss, not reclassified."""
    return masked_task_loss(
        prediction,
        batch,
        config["loss_enabled"],
        config["coordinate_scale"],
        config["residual_cap"],
        config["visibility_reduction"],
    )


def configure_active_optimizer(model, config):
    """Visibility loss and optimizer membership share one configuration switch."""
    optimizer = config["optimizer"]
    if optimizer["type"] != "AdamW":
        raise ValueError("unsupported_active_optimizer_policy")
    return configure_training_parameters(
        model,
        visibility_enabled=config["loss_enabled"]["visibility"],
        lr=optimizer["lr"],
        weight_decay=optimizer["weight_decay"],
        betas=tuple(optimizer["betas"]),
    )
