"""Five-slot bounded CPU acceptance; full canvases exist only in the reference branch."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'landmarks'))
import lib.models  # The upstream package has an existing import cycle if datasets is first.
from lib.datasets.mamma_roi_pilot import MammaROIPilot, native_plan, project_erp
from lib.datasets.utils_eval import gen_trans_from_patch_cv

CONFIG = ROOT / 'configs/retrain/mamma_roi_pilot_20261013.json'
OUT = ROOT / 'outputs/acceptance/mamma_data_adapter_20261013'
SOURCE = Path('/home/yy/Repos/mamma/landmarks/lib/datasets/vitdet_dataset.py')
spec = importlib.util.spec_from_file_location('lib.datasets._frozen_original_vitdet', SOURCE)
ref_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref_mod
spec.loader.exec_module(ref_mod)


def full_reference(slot):
    roi = slot['raster']['roi_xyxy']
    w, h = slot['full_image_size_wh']
    x0, y0, x1, y1 = roi
    local = cv2.imread(slot['files']['image']['path'], cv2.IMREAD_COLOR)
    local_mask = cv2.imread(slot['files']['person_mask']['path'], cv2.IMREAD_GRAYSCALE)
    canvas = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    canvas[y0:y1, x0:x1] = local
    mask[y0:y1, x0:x1] = local_mask
    cfg = SimpleNamespace(data_cfg={'image_size': [384, 512]})
    item = ref_mod.ViTDetDataset(cfg, canvas, mask,
                                  np.asarray([slot['mamma_native_crop']['box_xyxy']], np.float32),
                                  fast_inference=True)[0]
    box = np.asarray(slot['mamma_native_crop']['box_xyxy'], np.float32)
    center = (box[:2] + box[2:]) / 2
    affine = gen_trans_from_patch_cv(center[0], center[1], *item['box_size'], 384, 512, 1., 0.)
    return item, affine


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    d = MammaROIPilot(CONFIG)
    cfg = json.loads(CONFIG.read_text())
    with np.load(cfg['reference_faces']) as ref:
        faces = ref['faces'].copy()
        uv_count = len(ref['loop_uv'])
    R = np.load(cfg['reference_r138'])
    results = []
    assert len(d.slots) == 6 and len(d) == 5
    missing = [d.slot_status(i) for i in range(6) if i not in d.ready]
    assert len(missing) == 1 and missing[0]['reason'] == 'outside_physical_domain_no_instance_box'
    s0 = d.slots[d.ready[0]]
    roi = s0['raster']['roi_xyxy']
    box = s0['mamma_native_crop']['box_xyxy']
    for bad_stored, bad_roi in [([1, 1], roi), (s0['raster']['stored_size_wh'], [0, 0, 1, 1])]:
        try:
            native_plan(box, bad_roi, bad_stored, s0['full_image_size_wh'])
            raise AssertionError('invalid ROI accepted')
        except ValueError:
            pass
    try:
        native_plan([roi[0], roi[1], roi[0]+2, roi[1]+2], roi,
                    s0['raster']['stored_size_wh'], s0['full_image_size_wh'])
        raise AssertionError('unsupported crop accepted')
    except ValueError:
        pass
    for j in range(len(d)):
        slot = d.slots[d.ready[j]]
        item = d[j]
        assert item['img'].shape == (3, 512, 384) and item['mask'].shape == (1, 512, 384)
        assert item['roi_offset_xy'].any()
        f = np.load(slot['source_asset']['body_faces'], mmap_mode='r')
        faces_match = bool(np.array_equal(f, faces))
        assert faces_match
        ref, ref_affine = full_reference(slot)
        delta = np.abs(item['img'] - ref['img'])
        mask_delta = np.abs(item['mask'] - ref['mask'])
        affine_delta = np.abs(item['affine_erp_to_model'] - ref_affine)
        restored = cv2.transform(item['xy_model'][None].astype(np.float64), item['affine_model_to_erp'])[0]
        restore_err = np.linalg.norm(restored - item['xy_erp'], axis=1)
        body = np.asarray(np.load(slot['source_asset']['body_vertices'], mmap_mode='r')[slot['asset_frame_index']], dtype=np.float64)
        scene = np.asarray(np.load(slot['scene_matrices_ref'], mmap_mode='r')[slot['scene_matrix_row'], slot['scene_matrix_person_index']], dtype=np.float64)
        with np.load(slot['files']['annotations']['path']) as ann:
            person_ix = ann['person_ids'].tolist().index(slot['person_id'])
            camera = ann['camera_to_world_erp']
            r_world = (R @ body) @ scene[:3, :3].T + scene[:3, 3]
            r_xy = project_erp(r_world, camera, *slot['full_image_size_wh'])
            r_diff = np.linalg.norm(r_xy - ann['xy'][person_ix], axis=1)
            r_saved_world_diff = np.max(np.linalg.norm(r_world - ann['world_densekp'][person_ix], axis=1))
        result = dict(slot=j, episode=slot['episode'], rig=slot['rig'], observation_id=slot['observation_id'],
                      person_id=slot['person_id'], source_frame=slot['source_frame_index'],
                      asset_frame=slot['asset_frame_index'], time_tick=slot['source_time_tick'],
                      body_faces_match_locked_head=faces_match, reference_loop_uv_count=uv_count,
                      roi_offset_xy=item['roi_offset_xy'].tolist(), parent_box_xyxy=item['parent_box_xyxy'].tolist(),
                      image_nonzero_diff=int(np.count_nonzero(delta)), image_max_diff=float(delta.max()),
                      image_mean_diff=float(delta.mean()), mask_nonzero_diff=int(np.count_nonzero(mask_delta)),
                      mask_max_diff=float(mask_delta.max()), affine_max_diff=float(affine_delta.max()),
                      restore_max_px=float(restore_err.max()), r138_reproject_max_px=float(r_diff.max()),
                      r138_world_max_m=float(r_saved_world_diff),
                      geometry_reliable=int(item['geometry_reliable'].sum()),
                      physical_domain=int(item['physical_domain'].sum()),
                      roi_coverage=int(item['roi_coverage'].sum()), crop_support=int(item['crop_support'].sum()),
                      coordinate_loss_valid=int(item['coordinate_loss_valid'].sum()),
                      visibility_loss_valid=int(item['visibility_loss_valid'].sum()),
                      contact_loss_valid=int(item['contact_loss_valid'].sum()),
                      floor_contact_loss_valid=int(item['floor_contact_loss_valid'].sum()))
        point_file = OUT / f'slot_{j}_512_points.npz'
        np.savez_compressed(point_file,
                            vertex_ids=item['vertex_ids'], xy_erp=item['xy_erp'],
                            xy_model=item['xy_model'], xy_target=item['xy_target'],
                            coordinate_loss_valid=item['coordinate_loss_valid'],
                            physical_domain=item['physical_domain'],
                            roi_coverage=item['roi_coverage'], crop_support=item['crop_support'],
                            visibility=item['visibility'],
                            visibility_loss_valid=item['visibility_loss_valid'],
                            affine_erp_to_model=item['affine_erp_to_model'],
                            affine_model_to_erp=item['affine_model_to_erp'],
                            source_frame=np.int64(item['source_frame']),
                            source_time_tick=np.int64(item['source_time_tick']),
                            asset_frame_index=np.int64(item['asset_frame_index']),
                            person_id=np.int64(slot['person_id']))
        result['point_file'] = point_file.name
        results.append(result)
        print(json.dumps(result))
        del item, ref, body, delta, mask_delta
        gc.collect()
    output = dict(schema='mamma_roi_pilot_acceptance_v1', status='completed',
                  python=sys.executable, adapter_module=sys.modules['lib.datasets.mamma_roi_pilot'].__file__,
                  frozen_reference_module=ref_mod.__file__, slots=results, no_box=missing,
                  elapsed_seconds=time.monotonic()-t0, integration_only=True)
    (OUT / 'checks.json').write_text(json.dumps(output, indent=2) + '\n')
    (OUT / 'sample_manifest.json').write_text(json.dumps(dict(
        inventory=cfg['inventory'], selected_observation_ids=[s['observation_id'] for s in d.slots],
        eligibility='integration_only', box_policy=cfg['box_policy'], augmentation='off'), indent=2)+'\n')

if __name__ == '__main__':
    main()
