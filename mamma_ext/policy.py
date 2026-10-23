"""Explicit active training input/task policy; read-only planning is safe by default."""
import json
from pathlib import Path

from .data import MammaSavedGTDataset
from .tasks import configure_training_parameters, masked_task_loss


def load_active_policy(path, manifest_override=None):
    cfg = json.loads(Path(path).read_text())
    expected={'model_heads','coordinate_scale','residual_cap','optimizer','schema',
              'input_config','crop_policy_version','augmentation','loss_enabled',
              'supervised_visibility','execution_authorized','label_qualification',
              'model_config','tensor_state'}
    if (set(cfg)!=expected or cfg.get('schema') != 'mamma_training_v1' or
        cfg.get('crop_policy_version') != 'mask_bbox_inclusive_v1' or
        cfg.get('augmentation') != 'off' or
        type(cfg.get('execution_authorized')) is not bool):
        raise ValueError('unsupported_active_training_policy')
    visibility = cfg.get('supervised_visibility')
    if (type(visibility) is not bool or
        cfg.get('loss_enabled') != dict(coordinate=True, visibility=visibility)):
        raise ValueError('task_loss_optimizer_mismatch')
    dataset = MammaSavedGTDataset(cfg['input_config'], manifest_override=manifest_override)
    return cfg, dataset


def active_task_loss(prediction, batch, cfg):
    """Apply the policy task switches; unknown labels have no loss contribution."""
    return masked_task_loss(prediction,batch,cfg['loss_enabled'],
                            cfg['coordinate_scale'],cfg['residual_cap'])


def configure_active_optimizer(model, cfg):
    """Keep optimizer membership tied to the active task switch."""
    if (cfg['optimizer']['type'] != 'AdamW' or
        cfg['loss_enabled']['visibility'] is not cfg['supervised_visibility']):
        raise ValueError('unsupported_active_optimizer_policy')
    return configure_training_parameters(model,supervised_visibility=cfg['supervised_visibility'],
                                         lr=cfg['optimizer']['lr'],
                                         weight_decay=cfg['optimizer']['weight_decay'],
                                         betas=tuple(cfg['optimizer']['betas']))
