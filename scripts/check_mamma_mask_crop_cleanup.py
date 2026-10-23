"""Bounded CPU acceptance for the active mask-derived MAMMA crop policy.

Exactly ten real Dataset image reads: five legacy and five new. No model forward.
"""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acceptance/mamma_mask_derived_crop_cleanup_20261022'
PYTHON_MODULE=Path('/home/yy/Repos/riding-data/mamma_mask_roi.py')
sys.path.insert(0,str(ROOT/'landmarks'))
import lib.models  # noqa: F401; upstream import order, does not construct model
from lib.baseline.mamma_original import restore_xy
from lib.datasets.mamma_roi_pilot import (MammaManifestDataset, MammaMaskDerivedDataset,
                                          mask_box_inclusive, native_plan)
from lib.training.box_probe import VARIANTS, perturb_box
from lib.training.policy import load_active_policy
from lib.training.tasks import masked_task_loss


def rejects(fn,reason):
    try: fn()
    except ValueError as error:
        if str(error)!=reason: raise
        return reason
    raise AssertionError('expected_rejection_'+reason)


def synthetic_checks():
    spec=importlib.util.spec_from_file_location('deployment_mamma_mask_roi_read_only',PYTHON_MODULE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mask=np.zeros((6,7),np.uint8)
    mask[2:4,3:5]=255
    box=mask_box_inclusive(mask,(100,200))
    assert box.tolist()==[103,202,104,203]
    mask[2:4,5]=255
    changed=mask_box_inclusive(mask,(100,200))
    assert changed.tolist()==[103,202,105,203]
    empty=np.zeros((6,7),np.uint8)
    assert module.person_mask_roi(empty.astype(np.float32),(7,6)) is None
    empty_reason=rejects(lambda:mask_box_inclusive(empty,(100,200)),'empty_person_mask')
    one=empty.copy(); one[2,3]=255
    deployed_one=module.person_mask_roi(one.astype(np.float32)/255,(7,6))
    assert deployed_one is not None and deployed_one[2]==[3,2,3,2]
    one_reason=rejects(lambda:mask_box_inclusive(one,(100,200)),'degenerate_person_mask_box')
    return {'synthetic_box':box.tolist(),'changed_mask_box':changed.tolist(),
            'deployment_single_pixel_box':deployed_one[2],
            'active_single_pixel_rejection':one_reason,'empty_rejection':empty_reason,
            'deployment_empty_returns_none':True}


def inactive_visibility_check(active_cfg):
    batch={'xy_target':torch.tensor([[[0.,0.],[0.,0.]]]),
           'coordinate_loss_valid':torch.tensor([[True,True]]),
           'visibility':torch.tensor([[1,-1]]),
           'visibility_loss_valid':torch.tensor([[True,False]]),
           'contact':torch.tensor([[-1,-1]]),
           'contact_loss_valid':torch.tensor([[False,False]]),
           'floor_contact':torch.tensor([[-1,-1]]),
           'floor_contact_loss_valid':torch.tensor([[False,False]])}
    a={'joints2d':torch.tensor([[[1.,0.,0.],[1.,0.,0.]]]),
       'visibility':torch.tensor([[[12.],[-12.]]]),
       'contact':torch.zeros((1,2,1)), 'floor_contact':torch.zeros((1,2,1))}
    b={**a,'visibility':torch.tensor([[[-12.],[12.]]])}
    x=masked_task_loss(a,batch,active_cfg['loss_enabled'],coordinate_scale=1.)
    y=masked_task_loss(b,batch,active_cfg['loss_enabled'],coordinate_scale=1.)
    assert x['counts']['visibility']=={'points':1,'samples':1}
    assert float(x['parts']['visibility'])==float(y['parts']['visibility'])==0.
    assert float(x['loss'])==float(y['loss'])
    return {'known_positive_points':1,'visibility_loss_enabled':False,
            'visibility_loss':0.0,'total_loss_unchanged_with_opposite_visibility_logits':True}


def main():
    if (OUT/'checks.json').exists(): raise RuntimeError('existing_acceptance_refuses_overwrite')
    start=time.monotonic()
    active_cfg,new=load_active_policy(ROOT/'configs/retrain/mamma_training_active_20261022.json')
    old=MammaManifestDataset(ROOT/'configs/retrain/mamma_manifest_reader_20261015.json')
    assert len(new)==len(old)==5 and len(new.slots)==len(old.slots)==6
    assert new.slot_status(3)['reason']=='outside_physical_domain_no_instance_box'
    config_path=ROOT/'configs/retrain/mamma_mask_derived_reader_20261022.json'
    with TemporaryDirectory() as directory:
        path=Path(directory)/'reader.json'
        cfg=json.loads(config_path.read_text()); cfg['independent_box_augmentation']='enabled'; path.write_text(json.dumps(cfg))
        enabled_rejection=rejects(lambda:MammaMaskDerivedDataset(path),'mask_derived_reader_config_required')
        cfg.pop('independent_box_augmentation'); cfg['box_override']=[0,0,10,10]; path.write_text(json.dumps(cfg))
        override_config_rejection=rejects(lambda:MammaMaskDerivedDataset(path),'mask_derived_reader_config_required')
    override_call_rejection=rejects(lambda:new.get_with_box(0,[0,0,10,10]),'independent_box_override_disabled')
    synthetic=synthetic_checks()
    aux=inactive_visibility_check(active_cfg)
    old_smoke=json.loads((ROOT/'configs/retrain/mamma_training_smoke_20261015.json').read_text())
    assert old_smoke['loss_enabled']['visibility'] is True and active_cfg['loss_enabled']['visibility'] is False
    assert len(VARIANTS)==7 and perturb_box([1,2,11,22],'cx_plus_03').shape==(4,)
    rows=[]
    image_reads=0
    for i in range(5):
        old_item=old[i]; image_reads+=1
        item=new[i]; image_reads+=1
        box=item['parent_box_xyxy'].astype(np.int64)
        manifest=item['manifest_box_xyxy'].astype(np.int64)
        expected=manifest+np.array([0,0,-1,-1],np.int64)
        assert np.array_equal(box,expected), (i,box,manifest)
        assert item['box_source']=='stored_person_mask_cv2_boundingRect'
        assert item['crop_policy_version']=='mask_bbox_inclusive_v1'
        assert item['source_sampling_supported']
        assert not np.array_equal(item['affine_erp_to_model'],old_item['affine_erp_to_model'])
        center,size,sigma,rect=native_plan(box,item['roi_xyxy'],item['stored_size_wh'],item['full_size_wh'])
        assert np.array_equal(rect,item['sampling_rect_erp'])
        _,erp_box,erp_affine=restore_xy(item['xy_target'],item,True)
        valid=item['coordinate_loss_valid']
        erp_error=np.linalg.norm(erp_box[valid]-item['xy_erp'][valid],axis=1)
        affine_error=np.linalg.norm(erp_affine[valid]-item['xy_erp'][valid],axis=1)
        row={'slot':i,'identity':item['identity'],'manifest_box_xyxy':manifest.tolist(),
             'mask_box_xyxy_inclusive':box.tolist(),'mask_minus_manifest':(box-manifest).tolist(),
             'native_crop_box_exact':False,
             'image_input_different_elements':int(np.count_nonzero(item['img']!=old_item['img'])),
             'mask_input_different_elements':int(np.count_nonzero(item['mask']!=old_item['mask'])),
             'normalized_image_max_abs_difference':float(np.max(np.abs(item['img']-old_item['img']))),
             'affine_max_abs_difference':float(np.max(np.abs(item['affine_erp_to_model']-old_item['affine_erp_to_model']))),
             'coordinate_valid_old':int(old_item['coordinate_loss_valid'].sum()),
             'coordinate_valid_new':int(valid.sum()),
             'coordinate_valid_changed_points':int(np.count_nonzero(valid!=old_item['coordinate_loss_valid'])),
             'source_sampling_supported':True,'sampling_rect_erp':rect.tolist(),
             'sigma':float(sigma),'box_center':center.tolist(),'box_size':size.tolist(),
             'roi_offset_xy':item['roi_offset_xy'].tolist(),
             'erp_restore_box_max_px':float(erp_error.max()) if len(erp_error) else None,
             'erp_restore_affine_max_px':float(affine_error.max()) if len(affine_error) else None,
             'visibility_valid_points':int(item['visibility_loss_valid'].sum())}
        rows.append(row)
    result={'schema':'mamma_mask_derived_crop_acceptance_v1','crop_policy_version':'mask_bbox_inclusive_v1',
            'synthetic':synthetic,'normal_override_call_rejection':override_call_rejection,
            'enabled_independent_box_config_rejection':enabled_rejection,
            'override_config_rejection':override_config_rejection,
            'historical_diagnostic_import_and_perturb':True,
            'active_visibility':aux,'legacy_smoke_visibility_enabled':True,
            'absent_slot':new.slot_status(3),'samples':rows,
            'real_dataset_image_reads':image_reads,
            'cpu_seconds':time.monotonic()-start,
            'model_forwards':0,'optimizer_steps':0,'gpu_calls':0,'renders':0,
            'legacy_resume_contract':'mamma_infrastructure_resume_v1 keeps box_stream, off-only'}
    (OUT/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'samples':len(rows),'real_dataset_image_reads':image_reads,
                      'cpu_seconds':result['cpu_seconds'],
                      'boxes':[r['mask_box_xyxy_inclusive'] for r in rows],
                      'valid_points':[r['coordinate_valid_new'] for r in rows]},ensure_ascii=False))


if __name__=='__main__': main()
