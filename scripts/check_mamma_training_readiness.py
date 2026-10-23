"""Targeted CPU checks. Never constructs MammaNet or calls optimizer.step."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'landmarks'))
import lib.models  # noqa: F401; original package initialization order
from lib.datasets.mamma_roi_pilot import MammaROIPilot, MammaManifestDataset, native_plan
from lib.training.tasks import masked_task_loss
from lib.training.box_probe import preflight

OUT = ROOT/'outputs/experiments/mamma_training_readiness_box_20261015'


def check_losses():
    pred = {'joints2d': torch.tensor([[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]], requires_grad=True),
            'visibility': torch.zeros((1,3,1), requires_grad=True),
            'contact': torch.zeros((1,3,1)), 'floor_contact': torch.zeros((1,3,1))}
    batch = {'xy_target': torch.tensor([[[0., 0.], [float('nan'), 0.], [0., 0.]]]),
             'coordinate_loss_valid': torch.tensor([[True, False, True]]),
             'visibility': torch.tensor([[1, -1, 0]]),
             'visibility_loss_valid': torch.tensor([[True, False, True]]),
             'contact': torch.full((1,3), -1), 'contact_loss_valid': torch.zeros((1,3),dtype=torch.bool),
             'floor_contact': torch.full((1,3), -1), 'floor_contact_loss_valid': torch.zeros((1,3),dtype=torch.bool)}
    enabled = dict(coordinate=True, visibility=True, contact=False, floor_contact=False)
    result = masked_task_loss(pred,batch,enabled,coordinate_scale=1.)
    expected = (1./2 + 9./2)/2 + np.log(2)
    assert abs(float(result['loss'])-expected) < 1e-6, (result,expected)
    assert result['counts']['coordinate']=={'points':2,'samples':1}
    result['loss'].backward()
    assert torch.equal(pred['joints2d'].grad[0,1], torch.zeros(3))
    assert torch.equal(pred['visibility'].grad[0,1], torch.zeros(1))
    whole = {**batch, 'xy_target':torch.zeros((1,3,2)),
             'coordinate_loss_valid':torch.ones((1,3),dtype=torch.bool)}
    assert masked_task_loss(pred,whole,enabled)['counts']['coordinate']['points']==3
    empty = {**batch, 'coordinate_loss_valid':torch.zeros((1,3),dtype=torch.bool),
             'visibility_loss_valid':torch.zeros((1,3),dtype=torch.bool)}
    no = masked_task_loss(pred,empty,enabled)
    assert no['skip_optimizer'] and no['skip_reason']=='all_enabled_tasks_without_valid_points'
    assert float(no['loss'])==0 and no['counts']['coordinate']['samples']==0
    return {'partial_loss':float(result['loss']), 'expected_partial_loss':float(expected),
            'unknown_gradient_zero':True, 'full_valid':True, 'all_invalid_skipped':True}


def main():
    t=time.monotonic()
    if (OUT/'cpu_checks.json').exists():
        raise RuntimeError('existing_cpu_checks_refuses_overwrite')
    old = MammaROIPilot(ROOT/'configs/retrain/mamma_roi_pilot_20261013.json')
    new = MammaManifestDataset(ROOT/'configs/retrain/mamma_manifest_reader_20261015.json')
    assert len(old)==len(new)==5 and len(new.slots)==6
    assert new.slot_status(3)['reason']=='outside_physical_domain_no_instance_box'
    checks=[]
    for i in range(5):
        a,b=old[i],new[i]
        for key in ('img','mask','affine_erp_to_model','affine_model_to_erp',
                    'coordinate_loss_valid','coverage_crop'):
            assert np.array_equal(a[key],b[key],equal_nan=True), (i,key)
        geometry_delta={key:float(np.nanmax(np.abs(a[key]-b[key]))) for key in ('xy_erp','xy_model','xy_target')}
        assert max(geometry_delta.values()) < 1e-9, (i,geometry_delta)
        assert np.array_equal(b['physical_valid_crop'],b['coverage_crop'])
        assert np.array_equal(b['point_in_crop'],b['crop_support'])
        assert b['source_sampling_supported']
        assert not b['visibility_loss_valid'].any() and not b['contact_loss_valid'].any() and not b['floor_contact_loss_valid'].any()
        checks.append({'slot':i,'valid_points':int(b['coordinate_loss_valid'].sum()),
                       'image_mask_affine_exact':True,'geometry_max_delta':geometry_delta})
    s=new.slots[new.ready[0]]
    try: native_plan([1,1,0,0],s['raster']['roi_xyxy'],s['raster']['stored_size_wh'],s['full_image_size_wh'])
    except ValueError as error: assert str(error)=='invalid_box'
    else: raise AssertionError('invalid_box_not_rejected')
    try: native_plan(s['mamma_native_crop']['box_xyxy'],[0,0,1,1],[2,2],s['full_image_size_wh'])
    except ValueError as error: assert str(error)=='invalid_roi_size'
    else: raise AssertionError('invalid_roi_not_rejected')
    rows,items,common=preflight(new)
    assert len(rows)==35 and len(common)==5
    (OUT/'box_support.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'schema':'mamma_training_cpu_checks_v1','native_regression':checks,'absent_slot':new.slot_status(3),
            'loss':check_losses(),'box_support':{'planned':35,'supported':sum(r['status']=='supported' for r in rows),
                           'rejected':sum(r['status']=='rejected' for r in rows),
                           'common_valid_points':[int(common[i].sum()) for i in range(5)]},
            'seconds':time.monotonic()-t,'imports':{'dataset':str(Path(sys.modules['lib.datasets.mamma_roi_pilot'].__file__).resolve()),
                                                     'tasks':str(Path(sys.modules['lib.training.tasks'].__file__).resolve())}}
    (OUT/'cpu_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
