"""Bounded CPU checks for the 2026-10-22 simple training contract.

Reads exactly one old and one A3 observed Dataset item. No MammaNet or step.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'landmarks'))
import lib.models  # noqa: F401; upstream package initialization order
from lib.datasets.mamma_roi_pilot import MammaMaskDerivedDataset, parse_point_label
from lib.training.policy import load_active_policy, active_task_loss, configure_active_optimizer


def expect_error(fn, fragment):
    try:
        fn()
    except ValueError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('expected rejection: ' + fragment)


def tiny_model():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)
            self.mask_downscaling = torch.nn.Linear(1, 1)
            self.decoder = torch.nn.Module()
            for name in ('query_embed', 'decoder_detr', 'landmarks', 'vis_prob',
                         'contact_prob', 'floor_contact_prob'):
                setattr(self.decoder, name, torch.nn.Linear(1, 1))
    return Tiny()


def main():
    start = time.monotonic()
    out = ROOT / 'outputs/acceptance/mamma_simple_training_contract_20261022/checks.json'
    if out.exists():
        raise FileExistsError(out)
    old = MammaMaskDerivedDataset(ROOT/'configs/retrain/mamma_mask_derived_reader_20261022.json')
    cfg, a3 = load_active_policy(ROOT/'configs/retrain/mamma_training_simple_20261022.json')
    old_cfg, _ = load_active_policy(ROOT/'configs/retrain/mamma_training_active_20261022.json')
    assert len(old) == 5 and len(a3) == 10 and len(a3.slots) == 12
    assert old_cfg['loss_enabled']['visibility'] is False
    assert sum(s['status'] != 'observed' for s in a3.slots) == 2
    expect_error(lambda: a3.get_with_box(0, [0, 0, 20, 20]), 'independent_box_override_disabled')
    reads = 0
    samples = []
    for name, dataset in (('old_unknown', old), ('a3_positive', a3)):
        item = dataset[0]
        reads += 1
        assert item['img'].shape == (3, 512, 384)
        assert item['mask'].shape == (1, 512, 384)
        assert item['source_sampling_supported'] and item['source_filter_support_valid']
        assert item['box_source'] == 'stored_person_mask_cv2_boundingRect'
        assert np.array_equal(item['coordinate_loss_valid'],
                              item['geometry_reliable'] & item['physical_domain'] &
                              item['point_in_render_coverage'] & item['point_in_crop'])
        assert np.all(item['visibility_loss_valid'] <= item['coordinate_loss_valid'])
        xy = item['xy_model'] @ item['affine_model_to_erp'][:, :2].T + item['affine_model_to_erp'][:, 2]
        err = np.max(np.abs(xy[item['geometry_reliable']] - item['xy_erp'][item['geometry_reliable']]))
        assert err < 1e-9, err
        if name == 'old_unknown':
            assert not item['visibility_loss_valid'].any() and np.all(item['visibility'] == -1)
        else:
            assert np.count_nonzero(item['visibility_loss_valid']) > 0
            assert np.count_nonzero(item['visibility'] == 0) == 0
        assert np.all(item['contact'] == -1) and not item['contact_loss_valid'].any()
        assert np.all(item['floor_contact'] == -1) and not item['floor_contact_loss_valid'].any()
        samples.append({'name':name,'observation_id':item['identity']['observation_id'],
                        'box_xyxy':item['parent_box_xyxy'].tolist(),
                        'coordinate_valid':int(item['coordinate_loss_valid'].sum()),
                        'visibility_positive_valid':int((item['visibility_loss_valid'] & (item['visibility'] == 1)).sum()),
                        'visibility_negative_valid':int((item['visibility_loss_valid'] & (item['visibility'] == 0)).sum()),
                        'erp_roundtrip_max_abs_px':float(err)})

    region = np.ones(512, dtype=bool)
    vals = np.full(512, -1, dtype=np.int8); vals[:3] = [1, 0, -1]
    tri, valid = parse_point_label(vals, 'visibility', region)
    assert tri[:3].tolist() == [1, 0, -1] and valid[:3].tolist() == [True, True, False]
    legacy, legacy_valid = parse_point_label({'value':vals.tolist(), 'known':[False]*512,
                                              'evidence_ref':'legacy_reference'}, 'visibility', region)
    assert np.all(legacy == -1) and not legacy_valid.any()
    expect_error(lambda: parse_point_label({'value':vals.tolist(), 'known':[True]*512,
                                             'evidence_ref':'legacy_reference'}, 'visibility', region),
                 'point_label_known_unknown_conflict')
    expect_error(lambda: parse_point_label({'value':vals.tolist(), 'known':[False]*512,
                                             'valid':[True]*512}, 'visibility', region),
                 'point_label_valid_unknown_conflict')
    tri_clipped, valid_clipped = parse_point_label(vals, 'visibility',
                                                   np.r_[False, True, np.zeros(510, dtype=bool)])
    assert tri_clipped[0] == 1 and valid_clipped[:3].tolist() == [False, True, False]

    # Two valid visibility labels contribute to BCE. Unknown has no gradient.
    joints = torch.tensor([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]], requires_grad=True)
    logits = torch.tensor([[0.2,-0.4,2.0]], requires_grad=True)
    batch = {'xy_target':torch.zeros((1,3,2)),
             'coordinate_loss_valid':torch.tensor([[True,True,True]]),
             'visibility':torch.tensor([[1,0,-1]]),
             'visibility_loss_valid':torch.tensor([[True,True,False]])}
    pred = {'joints2d':joints,'visibility':logits}
    result = active_task_loss(pred,batch,cfg)
    result['loss'].backward()
    assert result['counts']['visibility'] == {'points':2,'samples':1}
    assert logits.grad[0,0] != 0 and logits.grad[0,1] != 0 and logits.grad[0,2] == 0
    assert joints.grad is not None and torch.isfinite(joints.grad).all()
    coordinate_loss = float(result['parts']['coordinate'].detach())
    for value in (1,0,-1):
        changed = dict(batch, visibility=torch.tensor([[value,value,value]]),
                       visibility_loss_valid=torch.tensor([[value != -1]*3]))
        got = active_task_loss(pred,changed,cfg)
        assert float(got['parts']['coordinate'].detach()) == coordinate_loss
    unknown = dict(batch,visibility=torch.full((1,3),-1),
                   visibility_loss_valid=torch.zeros((1,3),dtype=torch.bool))
    skip_vis = active_task_loss(pred,unknown,cfg)
    assert skip_vis['counts']['visibility']['points'] == 0 and not skip_vis['skip_optimizer']
    empty = dict(unknown,coordinate_loss_valid=torch.zeros((1,3),dtype=torch.bool))
    all_skip = active_task_loss(pred,empty,cfg)
    assert all_skip['skip_optimizer'] and all_skip['skip_reason'] == 'all_enabled_tasks_without_valid_points'

    groups = {}
    for key, policy in (('on',cfg),('off',old_cfg)):
        model = tiny_model()
        _, summary = configure_active_optimizer(model,policy)
        active = {name for name,p in model.named_parameters() if p.requires_grad}
        assert any(x.startswith('decoder.query_embed.') for x in active)
        assert any(x.startswith('decoder.decoder_detr.') for x in active)
        assert any(x.startswith('decoder.landmarks.') for x in active)
        assert any(x.startswith('decoder.vis_prob.') for x in active) == (key == 'on')
        assert not any(x.startswith(('backbone.','mask_downscaling.',
                                      'decoder.contact_prob.','decoder.floor_contact_prob.')) for x in active)
        groups[key] = [g['name'] for g in summary]
    report={'schema':'mamma_simple_training_contract_check_v1',
            'real_dataset_image_reads':reads,'samples':samples,
            'label_checks':'tri_state_legacy_unknown_conflict_and_region_pass',
            'loss_checks':'positive_negative_bce_unknown_zero_gradient_coordinate_independent_all_skip_pass',
            'parameter_groups':groups,'execution_authorized':False,
            'model_forwards':0,'optimizer_steps':0,'gpu_calls':0,'segmenter_calls':0,'renders':0,
            'cpu_seconds':round(time.monotonic()-start,3)}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
