"""Reusable fixed-input MammaNet checkpoint, restoration and metric helpers.

No detector, optimizer, download, dataset generation or training entry point.
"""

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from .upstream import crop_math


def deployment_config(path):
    """Mirror run_ma_2d.load_hydra_style_config without importing its dotenv CLI."""
    path = Path(path).resolve()
    for name, fn in {
        "mult": lambda x, y: x * y,
        "if": lambda x, y, z: y if x else z,
        "div": lambda x, y: x // y,
        "concat": lambda x: np.concatenate(x).tolist(),
        "sorted": lambda x: np.argsort(x).tolist(),
    }.items():
        OmegaConf.register_new_resolver(name, fn, replace=True)
    cfg = OmegaConf.load(path)
    base = OmegaConf.create()
    for entry in cfg.defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, Mapping):
            for group, name in entry.items():
                base[group] = OmegaConf.load(path.parent / group / f"{name}.yaml")
        else:
            base = OmegaConf.merge(base, OmegaConf.load(path.parent / f"{entry}.yaml"))
    del cfg["defaults"]
    merged = OmegaConf.merge(base, cfg)
    return OmegaConf.create(OmegaConf.to_container(merged, resolve=True, throw_on_missing=True))


def model_config_summary(cfg):
    decoder = cfg.model.decoder
    backbone = cfg.model.backbone
    return dict(
        model_name=cfg.model_name,
        num_joints=cfg.num_joints,
        image_size=list(cfg.data_cfg.image_size),
        normalize_plus_min_one=bool(cfg.data.train.normalize_plus_min_one),
        use_mask=bool(cfg.model.use_mask),
        freeze_backbone=bool(cfg.model.freeze_backbone),
        decoder=dict(
            layer_name=decoder.layer_name,
            n_landmarks=decoder.n_landmarks,
            uncertainty=decoder.uncertainty,
            visibility=decoder.visibility,
            contact=decoder.contact,
            floor_contact=decoder.floor_contact,
        ),
        backbone=dict(
            type=backbone.type,
            img_size=list(backbone.img_size),
            patch_size=backbone.patch_size,
            embed_dim=backbone.embed_dim,
            depth=backbone.depth,
        ),
    )


def strict_model_from_tensors(cfg, tensor_state_path):
    """Only tensor-only weights_only input; compare exact names/shapes before strict load."""
    from lib.models import build_model  # Heavy upstream import only in explicit model path.

    state = torch.load(tensor_state_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
    ):
        raise TypeError("tensor_state_not_string_tensor_map")
    model = build_model(cfg)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = {
        k: dict(model=list(expected[k].shape), checkpoint=list(state[k].shape))
        for k in expected.keys() & state.keys()
        if tuple(expected[k].shape) != tuple(state[k].shape)
    }
    report = dict(
        model_key_count=len(expected),
        checkpoint_key_count=len(state),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=shapes,
        prefix_conversion="none",
        strict_load=False,
        backbone_keys=sum(k.startswith("backbone.") for k in expected),
        checkpoint_backbone_keys=sum(k.startswith("backbone.") for k in state),
    )
    if missing or unexpected or shapes:
        return None, report
    incompatible = model.load_state_dict(state, strict=True)
    report["missing_keys"] = list(incompatible.missing_keys)
    report["unexpected_keys"] = list(incompatible.unexpected_keys)
    report["strict_load"] = True
    return model, report


def predict_one(model, item, device):
    img = torch.from_numpy(np.ascontiguousarray(item["img"][None])).to(
        device=device, dtype=torch.float32
    )
    mask = torch.from_numpy(np.ascontiguousarray(item["mask"][None])).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        output = model(img, mask)
    result = {
        key: value.detach().cpu().numpy().copy()
        for key, value in output.items()
        if value is not None
    }
    info = dict(
        input_image_shape=list(img.shape),
        input_mask_shape=list(mask.shape),
        input_dtype=str(img.dtype),
        output_shapes={k: list(v.shape) for k, v in result.items()},
        output_dtypes={k: str(v.dtype) for k, v in result.items()},
        nonfinite_counts={k: int((~np.isfinite(v)).sum()) for k, v in result.items()},
    )
    return result, info


def restore_xy(raw_xy, item, plus_minus_one=True):
    """Return deployment box restore and independently affine-inverted ERP xy."""
    raw = np.asarray(raw_xy, dtype=np.float64)
    normalized = (raw + 1) / 2 if plus_minus_one else raw.copy()
    box = np.asarray(item["parent_box_xyxy"], dtype=np.float32)
    center = (box[:2] + box[2:]) / 2
    # Exact deployment expansion follows the same saved affine metadata.
    expand_to_aspect_ratio, _ = crop_math()
    size = expand_to_aspect_ratio((box[2:] - box[:2]) * 1.2, (384, 512))
    model_xy = normalized * [384, 512]
    deployment_erp = normalized * size - size / 2 + center
    affine_erp = cv2.transform(model_xy[None], item["affine_model_to_erp"])[0]
    if item.get("periodic_x", False):
        width = int(item["full_size_wh"][0])
        deployment_erp[:, 0] %= width
        affine_erp[:, 0] %= width
    return model_xy, deployment_erp, affine_erp


def summarize_errors(pred_erp, gt_erp, valid, width=5760):
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(pred_erp).all(1)
    errors = np.full(len(valid), np.nan, dtype=np.float64)
    delta = pred_erp[valid & finite] - gt_erp[valid & finite]
    delta[:, 0] = (delta[:, 0] + width / 2) % width - width / 2
    errors[valid & finite] = np.linalg.norm(delta, axis=1)
    n = int(valid.sum())
    failed = int((valid & ~finite).sum())
    good = errors[valid & finite]
    # A nonfinite prediction remains in the denominator: the complete mean is infinite.
    return errors, dict(
        valid_points=n,
        nonfinite_prediction_points=failed,
        finite_evaluated_points=len(good),
        mean_erp_px=float("inf") if failed else float(good.mean()) if len(good) else None,
        median_erp_px=float(np.median(good)) if len(good) else None,
        p90_erp_px=float(np.percentile(good, 90)) if len(good) else None,
    )
