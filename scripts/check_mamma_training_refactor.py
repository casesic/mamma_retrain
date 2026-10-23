"""Bounded CPU acceptance for the project-owned regular MAMMA extension."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamma_ext.data import MammaMaskDerivedDataset, MammaSavedGTDataset, native_plan
from mamma_ext.recovery import regular_payload
from mamma_ext.runner import main as cli_main, visibility_metrics
from mamma_ext.split import validate_disjoint
import mamma_ext.runner as runner

OUT = ROOT/'outputs/acceptance/mamma_training_refactor_a1_20261022'


def rejects(fn, fragment):
    try:
        fn()
    except ValueError as exc:
        if fragment not in str(exc):
            raise AssertionError((fragment,str(exc))) from exc
    else:
        raise AssertionError('expected_rejection_'+fragment)


def main():
    start=time.monotonic()
    output=OUT/'checks.json'
    if output.exists():
        raise FileExistsError(output)
    old=MammaMaskDerivedDataset(ROOT/'configs/retrain/mamma_a3_mask_reader_20261022.json')
    new=MammaSavedGTDataset(ROOT/'configs/retrain/mamma_saved_gt_reader.json')
    assert len(old)==len(new)==10 and len(new.slots)==12
    assert 'lib.models' not in sys.modules, 'pure_dataset_imported_upstream_model_package'
    reads=0; comparisons=[]; blocked_gt_loads=[]
    for person_id in (1,2):
        row=next(s for s in new.slots if s['status']=='observed' and
                 s['camera_id']=='cam2' and s['frame_index']==0 and s['person_id']==person_id)
        key=(row['episode'],row['rig'],row['frame_index'],row['camera_id'],person_id)
        index=new.observation_index(*key)
        assert old.observation_index(*key)==index
        legacy=old[index];reads+=1
        original_load=np.load
        def guarded_load(path,*args,**kwargs):
            value=str(path)
            if value.endswith(('vertices.npy','faces.npy','scene_matrices.npy')):
                blocked_gt_loads.append(value)
                raise AssertionError('saved_gt_hot_path_loaded_body_or_scene:'+value)
            return original_load(path,*args,**kwargs)
        np.load=guarded_load
        try:
            direct=new[index];reads+=1
        finally:
            np.load=original_load
        for field in ('img','mask','affine_erp_to_model','affine_model_to_erp','parent_box_xyxy'):
            assert np.array_equal(legacy[field],direct[field]),(person_id,field)
        assert direct['gt_source']=='saved_mamma_512_npz_v1'
        assert direct['source_sampling_supported']
        assert np.array_equal(legacy['vertex_ids'],direct['vertex_ids'])
        assert np.array_equal(legacy['visibility'],direct['visibility'])
        delta=np.abs(legacy['xy_erp']-direct['xy_erp'])
        finite=np.isfinite(delta).all(1)
        comparisons.append({'person_id':person_id,'observation_id':row['observation_id'],
            'xy_max_abs_px':float(delta[finite].max()),
            'xy_mean_abs_px':float(delta[finite].mean()),
            'coordinate_valid_old':int(legacy['coordinate_loss_valid'].sum()),
            'coordinate_valid_saved':int(direct['coordinate_loss_valid'].sum()),
            'coordinate_valid_changed':int(np.count_nonzero(
                legacy['coordinate_loss_valid']!=direct['coordinate_loss_valid'])),
            'visibility_valid_changed':int(np.count_nonzero(
                legacy['visibility_loss_valid']!=direct['visibility_loss_valid'])),
            'physical_mask_saved_disagreement':int(direct['physical_mask_disagreement'].sum()),
            'roi_saved_disagreement':int(direct['coverage_disagreement'].sum()),
            'visibility_known_positive':int(np.count_nonzero(direct['visibility_loss_valid'] &
                                                             (direct['visibility']==1))),
            'visibility_known_negative':int(np.count_nonzero(direct['visibility_loss_valid'] &
                                                             (direct['visibility']==0)))})
    assert not blocked_gt_loads
    absent=next((i,s) for i,s in enumerate(new.slots) if s['status']!='observed')
    slot_index,slot=absent
    assert new.slot_status(slot_index)['reason']==slot['reason']
    rejects(lambda:new.observation_index(slot['episode'],slot['rig'],slot['frame_index'],
                                         slot['camera_id'],slot['person_id']), 'observation_absent:')
    rejects(lambda:new.get_with_box(0,[0,0,10,10]), 'independent_box_override_disabled')
    rejects(lambda:native_plan([10,10,20,20],[0,0,24,24],[24,24],[100,100]),
            'crop_filter_support_outside_roi')

    # A one-row manifest keeps each rejection small; A3 source files stay read-only.
    with tempfile.TemporaryDirectory(dir=OUT,prefix='fixture_') as tmp:
        base=Path(tmp)
        original_manifest=json.loads(Path(new.cfg['manifest']).read_text())
        source=copy.deepcopy(original_manifest['observations'][0])
        def configured(changed, name):
            manifest=dict(original_manifest,observations=[changed])
            mp=base/(name+'_manifest.json');mp.write_text(json.dumps(manifest))
            cfg=dict(new.cfg,manifest=str(mp))
            cp=base/(name+'_reader.json');cp.write_text(json.dumps(cfg))
            return cp
        wrong=copy.deepcopy(source);wrong['person_id']=999
        rejects(lambda:MammaSavedGTDataset(configured(wrong,'person')),'saved_gt_person_mismatch')
        wrong=copy.deepcopy(source);wrong['source_time_tick']+=1
        rejects(lambda:MammaSavedGTDataset(configured(wrong,'time')),
                'saved_gt_identity_time_camera_mismatch')
        wrong=copy.deepcopy(source);wrong['camera_matrix_ref']='wrong:camera_to_world_erp'
        rejects(lambda:MammaSavedGTDataset(configured(wrong,'camera')),
                'saved_gt_camera_reference_mismatch')
        cam=base/'cam2';cam.mkdir()
        gt_path=cam/'mamma_512.npz'; ann_path=cam/'annotations.npz'
        with np.load(source['files']['mamma_512']['path'],allow_pickle=False) as original:
            arrays={k:original[k] for k in original.files}
        ids=arrays['ordered_vertex_ids'].copy();ids[[0,1]]=ids[[1,0]]
        arrays['ordered_vertex_ids']=ids
        np.savez_compressed(gt_path,**arrays)
        shutil.copyfile(source['files']['annotations']['path'],ann_path)
        wrong=copy.deepcopy(source)
        wrong['files']['mamma_512']['path']=str(gt_path)
        wrong['files']['annotations']['path']=str(ann_path)
        wrong['camera_matrix_ref']=str(ann_path)+':camera_to_world_erp'
        for field,ref in wrong['mamma_512'].items():
            if field.endswith('_ref'):
                wrong['mamma_512'][field]=ref.replace(source['files']['mamma_512']['path'],str(gt_path))
        wrong['point_labels']['visibility']['evidence_ref']=str(gt_path)
        rejects(lambda:MammaSavedGTDataset(configured(wrong,'order')),
                'saved_gt_template_order_mismatch')
        wrong=copy.deepcopy(source);wrong['point_labels']['visibility']['value'][0]=1
        rejects(lambda:MammaSavedGTDataset(configured(wrong,'label')),
                'saved_gt_visibility_manifest_conflict')

        # Current activity refuses before importing model code or reading weights.
        arguments=['train','--config',str(ROOT/'configs/retrain/mamma_regular.json'),
                   '--validation-manifest',new.cfg['manifest'],'--max-updates','1',
                   '--checkpoint-every','1','--output',str(OUT/'never_train'),
                   '--device','cpu','--execute']
        rejects(lambda:cli_main(arguments),'configuration_execution_not_authorized')
        rejects(lambda:cli_main(['evaluate','--config',str(ROOT/'configs/retrain/mamma_regular.json'),
            '--output',str(OUT/'never_eval'),'--device','cpu','--execute']),
            'configuration_execution_not_authorized')
        assert 'lib.models' not in sys.modules, 'rejected_execution_imported_model'
        # Mock only dispatch; no real train/evaluate implementation is called.
        cfg=json.loads((ROOT/'configs/retrain/mamma_regular.json').read_text())
        cfg['execution_authorized']=True;cfg['label_qualification']='qualified'
        cp=base/'mock_authorized_only.json';cp.write_text(json.dumps(cfg))
        seen=[]; original_train,original_evaluate=runner.train,runner.evaluate
        runner.train=lambda args,cfg,dataset:seen.append(('train',args.max_updates,len(dataset)))
        runner.evaluate=lambda args,cfg,dataset:seen.append(('evaluate',len(dataset)))
        try:
            train_args=arguments.copy();train_args[train_args.index('--config')+1]=str(cp)
            cli_main(train_args)
            cli_main(['evaluate','--config',str(cp),'--output',str(OUT/'mock_eval'),
                      '--device','cpu','--execute'])
        finally:
            runner.train,runner.evaluate=original_train,original_evaluate
        assert seen==[('train',1,10),('evaluate',10)]
    rejects(lambda:validate_disjoint(new,new),'train_validation_group_overlap_episodes')
    assert visibility_metrics([1,1],[3.,2.])['status']=='single_class_or_unlabeled_no_accuracy'
    assert visibility_metrics([1,0],[3.,-2.])['accuracy']==1.
    refs={k:'fixture' for k in ('config','train_manifest','validation_manifest','model_config',
                                'tensor_state','crop_policy_version','gt_policy','augmentation')}
    payload=regular_payload({'x':torch.ones(1)},{'state':{}},1,
                            {'torch_cpu':torch.get_rng_state(),'torch_cuda':[]},
                            {'epoch':0,'next_position':1},refs)
    buf=io.BytesIO();torch.save(payload,buf);buf.seek(0)
    restored=torch.load(buf,map_location='cpu',weights_only=True)
    assert restored['schema']=='mamma_regular_resume_v2' and restored['global_update']==1
    assert restored['sampler']=={'epoch':0,'next_position':1} and restored['refs']==refs

    # Legacy imports remain aliases. Importing the upstream model package is only
    # for this compatibility check; no model is constructed.
    sys.path.insert(0,str(ROOT/'landmarks'))
    import lib.models  # noqa: F401
    from lib.datasets.mamma_roi_pilot import MammaSavedGTDataset as LegacyReader
    from lib.training.tasks import masked_task_loss as LegacyLoss
    from lib.baseline.mamma_original import summarize_errors as LegacyErrors
    from mamma_ext.tasks import masked_task_loss
    from mamma_ext.model import summarize_errors
    assert LegacyReader is MammaSavedGTDataset and LegacyLoss is masked_task_loss
    assert LegacyErrors is summarize_errors

    result={'schema':'mamma_training_refactor_acceptance_v1',
            'image_dataset_reads':reads,'compared_targets':comparisons,
            'hot_path_body_scene_reads':len(blocked_gt_loads),
            'fixture_rejections':['wrong_order','wrong_person','wrong_integer_time',
                                  'wrong_camera_ref','visibility_conflict','unsupported_crop',
                                  'independent_box','absent_target'],
            'legacy_imports':True,'cli_dispatch_mock':seen,
            'regular_resume_synthetic_roundtrip':True,
            'model_constructs':0,'model_forwards':0,'optimizer_steps':0,'gpu_calls':0,
            'segmenter_calls':0,'renders':0,'cpu_check_seconds':round(time.monotonic()-start,3)}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        failure=OUT/'failure.json'
        if not failure.exists():
            failure.write_text(json.dumps({'type':type(exc).__name__,'reason':str(exc)},
                                          ensure_ascii=False,indent=2)+'\n')
        raise
