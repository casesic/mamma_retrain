"""Validate the one current training policy and bind it to data and tasks."""

import json
from pathlib import Path

from .data import MammaSavedGTDataset
from .tasks import configure_training_parameters, masked_task_loss

TRAINING_SCHEMA = "mamma_training_v2"
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
    "execution_authorized",
    "label_qualification",
    "model_config",
    "tensor_state",
}


def _config_path(config_file, reference):
    """Project-owned references are relative to their declaring config file."""
    path = Path(reference)
    return (
        str((config_file.parent / path).resolve())
        if not path.is_absolute()
        else str(path.resolve())
    )


def load_active_policy(path, manifest_override=None):
    config_file = Path(path).resolve()
    config = json.loads(config_file.read_text())
    if (
        set(config) != POLICY_KEYS
        or config.get("schema") != TRAINING_SCHEMA
        or config.get("crop_policy_version") != "mask_bbox_inclusive_v1"
        or config.get("augmentation") != "off"
        or type(config.get("execution_authorized")) is not bool
        or config.get("model_heads") != ["joints2d", "visibility", "contact", "floor_contact"]
    ):
        raise ValueError("unsupported_active_training_policy")
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
    dataset = MammaSavedGTDataset(config["input_config"], manifest_override=manifest_override)
    return config, dataset


def active_task_loss(prediction, batch, config):
    """Unknown visibility labels are skipped by the task loss, not reclassified."""
    return masked_task_loss(
        prediction,
        batch,
        config["loss_enabled"],
        config["coordinate_scale"],
        config["residual_cap"],
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
