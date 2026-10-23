"""Historical 2026-10-15 diagnostic-only same-image parent-box probe.

This module is not an input path for the active mask-derived training reader.
"""
import numpy as np
from lib.baseline.mamma_original import restore_xy, summarize_errors


VARIANTS = ('native', 'cx_minus_03', 'cx_plus_03', 'cy_minus_03',
            'cy_plus_03', 'scale_09', 'scale_11')


def perturb_box(box, name):
    box = np.asarray(box, np.float32)
    center = (box[:2]+box[2:])/2
    size = box[2:]-box[:2]
    if name == 'cx_minus_03': center[0] -= .03*size[0]
    elif name == 'cx_plus_03': center[0] += .03*size[0]
    elif name == 'cy_minus_03': center[1] -= .03*size[1]
    elif name == 'cy_plus_03': center[1] += .03*size[1]
    elif name == 'scale_09': size *= .9
    elif name == 'scale_11': size *= 1.1
    elif name != 'native': raise ValueError('unknown_box_variant')
    return np.r_[center-size/2, center+size/2]


def preflight(dataset):
    """Freeze support and GT denominators before reading predictions."""
    rows, items, common = [], {}, {}
    for parent in range(len(dataset)):
        slot = dataset.slots[dataset.ready[parent]]
        native = slot['mamma_native_crop']['box_xyxy']
        for name in VARIANTS:
            box = perturb_box(native, name)
            key = (parent, name)
            row = {'parent': parent, 'variant': name, 'box_xyxy': box.tolist(),
                   'delta_center_fraction': [float((box[:2]+box[2:]-np.asarray(native[:2])-np.asarray(native[2:]))[i]/(2*(native[i+2]-native[i]))) for i in range(2)],
                   'scale': float((box[2]-box[0])/(native[2]-native[0]))}
            try:
                item = dataset.get_with_box(parent, box)
                if not item['source_sampling_supported']:
                    raise ValueError('source_sampling_unsupported')
                items[key] = item
                row.update(status='supported', valid_points=int(item['coordinate_loss_valid'].sum()),
                           sampling_rect_erp=item['sampling_rect_erp'].tolist())
            except (ValueError, IndexError, OSError) as error:
                row.update(status='rejected', reason=str(error))
            rows.append(row)
        masks = [items[(parent, n)]['coordinate_loss_valid'] for n in VARIANTS if (parent, n) in items]
        common[parent] = np.logical_and.reduce(masks) if masks else np.zeros(512, bool)
    for row in rows:
        row['common_valid_points'] = int(common[row['parent']].sum()) if row['status']=='supported' else 0
    return rows, items, common


def score_prediction(raw_joints2d, item, common_valid, native_erp):
    model_xy, erp, affine_erp = restore_xy(raw_joints2d[:, :2], item, True)
    restoration_delta = np.linalg.norm(erp-affine_erp, axis=1)
    if not np.isfinite(restoration_delta).all() or restoration_delta.max() > .01:
        raise ValueError('deployment_affine_restoration_disagree')
    _, own = summarize_errors(erp, item['xy_erp'], item['coordinate_loss_valid'])
    _, shared = summarize_errors(erp, item['xy_erp'], common_valid)
    drift = np.linalg.norm(erp[common_valid] - native_erp[common_valid], axis=1)
    return {'own': own, 'common': shared,
            'common_drift_mean_erp_px': float(drift.mean()) if len(drift) else None,
            'common_drift_median_erp_px': float(np.median(drift)) if len(drift) else None,
            'common_drift_p90_erp_px': float(np.percentile(drift,90)) if len(drift) else None,
            'restore_max_difference_px': float(restoration_delta.max())}, model_xy, erp
