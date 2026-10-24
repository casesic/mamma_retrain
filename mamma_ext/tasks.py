"""Explicit task validity and optimizer membership for qualified MAMMA observations."""

import torch
import torch.nn.functional as F

TASKS = ("coordinate", "visibility")


def masked_task_loss(pred, batch, enabled, coordinate_scale=100.0, residual_cap=25.0):
    """Per-sample valid-weight mean, then mean over labeled samples.

    Input arrays are batched tensors. Unknown -1 and invalid NaNs are never indexed.
    The old residual order is weight * squared distance / (2 sigma^2), clip,
    then add weight * 2 log sigma. Here all point weights are one.
    """
    anchor = pred["joints2d"]
    zero = anchor.new_zeros(())
    losses = {}
    counts = {}
    for task in TASKS:
        valid_key = "coordinate_loss_valid" if task == "coordinate" else "visibility_loss_valid"
        if valid_key not in batch:
            raise ValueError("task_validity_missing_" + task)
        valid = batch[valid_key].bool()
        if valid.ndim != 2 or valid.shape != anchor.shape[:2]:
            raise ValueError("invalid_task_mask_shape_" + task)
        count = valid.sum(1)
        counts[task] = {"points": int(count.sum()), "samples": int((count > 0).sum())}
        if not enabled[task] or not torch.any(valid):
            losses[task] = zero
            continue
        output = anchor if task == "coordinate" else pred["visibility"]
        if output is None:
            raise ValueError("enabled_head_missing_" + task)
        target = batch["xy_target"] if task == "coordinate" else batch[task]
        per_sample = []
        for b in range(valid.shape[0]):
            selected = valid[b]
            if not torch.any(selected):
                continue
            if task == "coordinate":
                p = output[b, selected]
                gt = target[b, selected]
                if not torch.isfinite(gt).all() or not torch.isfinite(p).all():
                    raise ValueError("nonfinite_valid_coordinate")
                log_sigma = torch.clamp(p[:, 2], min=torch.log(p.new_tensor(1e-6)))
                residual = ((gt - p[:, :2]) ** 2).sum(-1) / (2 * torch.exp(log_sigma).square())
                values = torch.clamp(residual, max=residual_cap) + 2 * log_sigma
                per_sample.append(values.mean() * coordinate_scale)
            else:
                label = target[b, selected].float()
                logits = output[b, selected].reshape(-1)
                if not torch.isin(label, torch.tensor([0.0, 1.0], device=label.device)).all():
                    raise ValueError("invalid_known_label_" + task)
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite_valid_logit_" + task)
                bce = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
                per_sample.append(bce.mean())
        losses[task] = torch.stack(per_sample).mean()
    active = [task for task in TASKS if enabled[task] and counts[task]["points"]]
    total = sum((losses[task] for task in active), zero)
    return {
        "loss": total,
        "parts": losses,
        "counts": counts,
        "skip_optimizer": not active,
        "skip_reason": "all_enabled_tasks_without_valid_points" if not active else None,
    }


def configure_training_parameters(
    model, visibility_enabled=False, lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999)
):
    """Freeze feature encoders and unsupervised heads; expose named AdamW groups."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = {
        "queries": "decoder.query_embed.",
        "shared_decoder": "decoder.decoder_detr.",
        "landmarks": "decoder.landmarks.",
    }
    if visibility_enabled:
        names["visibility"] = "decoder.vis_prob."
    groups = []
    grouped = set()
    for group_name, prefix in names.items():
        selected = [
            (name, param) for name, param in model.named_parameters() if name.startswith(prefix)
        ]
        if not selected:
            raise ValueError("missing_parameter_group_" + group_name)
        for name, param in selected:
            param.requires_grad_(True)
            if name in grouped:
                raise ValueError("duplicate_parameter_" + name)
            grouped.add(name)
        groups.append(
            {
                "name": group_name,
                "params": [param for _, param in selected],
                "lr": lr,
                "weight_decay": weight_decay,
                "parameter_names": [name for name, _ in selected],
            }
        )
    if any(
        name.startswith(
            (
                "backbone.",
                "mask_downscaling.",
                "decoder.contact_prob.",
                "decoder.floor_contact_prob.",
            )
        )
        and param.requires_grad
        for name, param in model.named_parameters()
    ):
        raise AssertionError("forbidden_parameter_unfrozen")
    optimizer = torch.optim.AdamW(
        [{k: v for k, v in g.items() if k != "parameter_names"} for g in groups], lr=lr, betas=betas
    )
    summary = [
        {
            "name": g["name"],
            "parameter_names": g["parameter_names"],
            "scalar_count": sum(p.numel() for p in g["params"]),
            "lr": lr,
            "weight_decay": weight_decay,
        }
        for g in groups
    ]
    return optimizer, summary


def set_training_mode(model):
    model.train()
    model.backbone.eval()
    model.mask_downscaling.eval()
    if model.backbone.training or model.mask_downscaling.training:
        raise AssertionError("frozen_feature_module_in_training_mode")
