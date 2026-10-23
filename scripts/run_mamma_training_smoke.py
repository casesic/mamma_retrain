"""Historical 2026-10-15 infrastructure smoke: three-step budget already exhausted.

Its dated config and checkpoint remain unchanged. Active input policy is elsewhere.
Default is plan only. Existing step logs block a rerun. Outputs are never deployable.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'landmarks'))
import lib.models  # noqa: F401
from lib.baseline.mamma_original import deployment_config, strict_model_from_tensors
from lib.datasets.mamma_roi_pilot import MammaManifestDataset
from lib.training.tasks import configure_training_parameters, masked_task_loss, set_training_mode
from lib.training.recovery import (capture_rng, clone_state,
                                   compare_state, save_update_boundary, load_update_boundary)

CFG_PATH=ROOT/'configs/retrain/mamma_training_smoke_20261015.json'
OUT=ROOT/'outputs/experiments/mamma_training_readiness_box_20261015'
CHECKPOINT=OUT/'infrastructure_smoke_update1_resume.pt'
ATTEMPTS=OUT/'optimizer_step_attempts.jsonl'
STEPS=OUT/'optimizer_steps.jsonl'


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def append_json(path,obj):
    with path.open('a') as file: file.write(json.dumps(obj,ensure_ascii=False)+'\n')


def tensor_batch(item,device):
    keys=('img','mask','xy_target','coordinate_loss_valid','visibility','visibility_loss_valid',
          'contact','contact_loss_valid','floor_contact','floor_contact_loss_valid')
    result={key:torch.as_tensor(np.ascontiguousarray(item[key][None]),device=device) for key in keys}
    result['img']=result['img'].float(); result['mask']=result['mask'].float()
    result['xy_target']=result['xy_target'].float()
    return result


def make_model(cfg,base,device):
    model,load=strict_model_from_tensors(cfg,base['tensor_state'])
    if model is None or not load['strict_load'] or load['model_key_count']!=282:
        raise RuntimeError('strict_original_282_key_load_failed')
    model.to(device)
    optimizer,groups=configure_training_parameters(model,supervised_visibility=False,lr=1e-5,weight_decay=1e-4)
    set_training_mode(model)
    return model,optimizer,load,groups


def perform_step(model,optimizer,batch,cfg,branch,logical_step,actual_step):
    set_training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    prediction=model(batch['img'],batch['mask'])
    loss_report=masked_task_loss(prediction,batch,cfg['loss_enabled'],cfg['coordinate_scale'],cfg['residual_cap'])
    if loss_report['skip_optimizer']: raise RuntimeError('unexpected_all_invalid_smoke_batch')
    loss=loss_report['loss']
    if not torch.isfinite(loss): raise RuntimeError('nonfinite_smoke_loss')
    loss.backward()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any(p.grad is None or not torch.isfinite(p.grad).all() for n,p in trainable):
        raise RuntimeError('missing_or_nonfinite_trainable_gradient')
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('backbone.','mask_downscaling.','decoder.vis_prob.','decoder.contact_prob.','decoder.floor_contact_prob.'))):
        raise RuntimeError('frozen_head_or_encoder_got_gradient')
    event={'branch':branch,'logical_step':logical_step,'actual_optimizer_step':actual_step,
           'loss_before_update':float(loss.detach().cpu()),'counts':loss_report['counts'],
           'loss_parts':{k:float(v.detach().cpu()) for k,v in loss_report['parts'].items()},
           'skip_reason':loss_report['skip_reason']}
    append_json(ATTEMPTS,{**event,'phase':'immediately_before_optimizer_step'})
    optimizer.step()
    append_json(STEPS,{**event,'phase':'optimizer_step_returned'})
    return event


def resume_remaining_step(cfg):
    """Continue after a process failure, never replay an already completed update."""
    if not CHECKPOINT.exists() or not STEPS.exists() or not ATTEMPTS.exists():
        raise RuntimeError('resume_artifacts_missing')
    steps=[json.loads(line) for line in STEPS.read_text().splitlines()]
    attempts=[json.loads(line) for line in ATTEMPTS.read_text().splitlines()]
    if len(steps)!=2 or len(attempts)!=2 or [(s['branch'],s['logical_step']) for s in steps]!=[('main',1),('main',2)]:
        raise RuntimeError('resume_refuses_unknown_step_history')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('PYTHON_DOTENV_DISABLED')!='1':
        raise RuntimeError('required_isolated_environment_missing')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('selected_gpu_unavailable')
    dataset=MammaManifestDataset(cfg['manifest_reader'])
    batch=tensor_batch(dataset[cfg['batch_indices'][1]],device)
    base=json.loads(Path(cfg['baseline_config']).read_text())
    model_cfg=deployment_config(base['model_config'])
    model,opt,load,groups=make_model(model_cfg,base,device)
    original=clone_state(model.state_dict())
    saved=load_update_boundary(CHECKPOINT,model,opt)
    if saved['global_update']!=1 or saved['sampler']['next_position']!=1 or saved['box_stream']['augmentation']!='off':
        raise RuntimeError('resume_boundary_mismatch')
    del saved
    event=perform_step(model,opt,batch,cfg,'resume',2,3)
    frozen_changes=[n for n,p in model.named_parameters() if not p.requires_grad and not torch.equal(original[n],p.detach().cpu())]
    allowed_changes=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(original[n],p.detach().cpu())]
    all_steps=[json.loads(line) for line in STEPS.read_text().splitlines()]
    comparison={'actual_optimizer_steps':len(all_steps),'next_loss_main':steps[1]['loss_before_update'],
                'next_loss_resume':event['loss_before_update'],
                'next_loss_exact':steps[1]['loss_before_update']==event['loss_before_update'],
                'model_optimizer_cross_branch_comparison':'unavailable_main_update2_snapshot_lost_after_exception',
                'global_update_main':2,'global_update_resume':2,
                'frozen_parameter_changed_count':len(frozen_changes),
                'allowed_parameter_changed_count':len(allowed_changes),
                'allowed_parameter_changed_examples':allowed_changes[:10],
                'checkpoint_bytes':CHECKPOINT.stat().st_size,'not_deployable':True,
                'eligibility':'infrastructure_smoke_only'}
    write_json(OUT/'smoke_resume_comparison.json',comparison)
    if len(all_steps)!=3 or frozen_changes or not allowed_changes or not comparison['next_loss_exact']:
        raise RuntimeError('resume_continuation_failed; see comparison')
    print(json.dumps(comparison))


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--execute',action='store_true')
    args.add_argument('--resume-only',action='store_true')
    args.add_argument('--steps',type=int)
    a=args.parse_args()
    cfg=json.loads(CFG_PATH.read_text())
    if cfg['schema']!='mamma_training_smoke_v1' or cfg['planned_logical_updates']!=2 or cfg['maximum_actual_optimizer_steps']!=3:
        raise RuntimeError('smoke_config_not_bounded')
    plan={'mode':'plan','batch_indices':cfg['batch_indices'],'logical_updates':2,
          'actual_optimizer_steps':3,'checkpoint_after':'update1','eligibility':'infrastructure_smoke_only',
          'loss_enabled':cfg['loss_enabled'],'augmentation':cfg['augmentation']}
    if a.resume_only:
        if not a.execute or a.steps!=2: raise RuntimeError('explicit_execute_steps_2_required_for_resume')
        resume_remaining_step(cfg)
        return
    if not a.execute:
        if not (OUT/'smoke_plan.json').exists():
            write_json(OUT/'smoke_plan.json',plan)
        print(json.dumps(plan)); return
    if a.steps!=2: raise RuntimeError('explicit_steps_2_required')
    if STEPS.exists() or ATTEMPTS.exists() or CHECKPOINT.exists():
        raise RuntimeError('existing_step_or_checkpoint_refuses_automatic_rerun')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('PYTHON_DOTENV_DISABLED')!='1':
        raise RuntimeError('required_isolated_environment_missing')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('selected_gpu_unavailable')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    started=time.monotonic()
    device=torch.device('cuda:0')
    dataset=MammaManifestDataset(cfg['manifest_reader'])
    if len(dataset)!=5 or any(dataset[i]['visibility_loss_valid'].any() or dataset[i]['contact_loss_valid'].any() or dataset[i]['floor_contact_loss_valid'].any() for i in cfg['batch_indices']):
        raise RuntimeError('smoke_sample_or_unknown_label_contract_changed')
    batches=[tensor_batch(dataset[i],device) for i in cfg['batch_indices']]
    base=json.loads(Path(cfg['baseline_config']).read_text())
    model_cfg=deployment_config(base['model_config'])
    model,opt,load,groups=make_model(model_cfg,base,device)
    original=clone_state(model.state_dict())
    write_json(OUT/'smoke_parameter_groups.json',{'groups':groups,'strict_load':load,
               'frozen_parameter_names':[n for n,p in model.named_parameters() if not p.requires_grad],
               'trainable_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad]})
    main_events=[]
    main_events.append(perform_step(model,opt,batches[0],cfg,'main',1,1))
    if time.monotonic()-started>1100: raise TimeoutError('gpu_budget_near_limit_after_update1')
    save_update_boundary(CHECKPOINT,model,opt,1,capture_rng(),
                         {'fixed_indices':cfg['batch_indices'],'next_position':1,'epoch':0},
                         {'augmentation':'off','epoch':0,'python_random_state':random.getstate()},
                         {'data_manifest':json.loads(Path(json.loads(Path(cfg['manifest_reader']).read_text())['manifest']).read_text())['schema'],
                          'data_config_ref':cfg['manifest_reader'],'smoke_config_ref':str(CFG_PATH),
                          'model_config_ref':base['model_config'],'tensor_state_ref':base['tensor_state']})
    main_events.append(perform_step(model,opt,batches[1],cfg,'main',2,2))
    main_model=clone_state(model.state_dict()); main_opt=clone_state(opt.state_dict())
    frozen_changes=[n for n,p in model.named_parameters() if not p.requires_grad and not torch.equal(original[n],p.detach().cpu())]
    allowed_changes=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(original[n],p.detach().cpu())]
    if frozen_changes or not allowed_changes: raise RuntimeError('frozen_changed_or_trainable_unchanged')
    del model,opt
    torch.cuda.empty_cache()
    if time.monotonic()-started>1100: raise TimeoutError('gpu_budget_near_limit_before_restore')
    model,opt,reload,groups2=make_model(model_cfg,base,device)
    restored=load_update_boundary(CHECKPOINT,model,opt)
    if restored['global_update']!=1 or restored['sampler']['next_position']!=1 or restored['box_stream']['augmentation']!='off':
        raise RuntimeError('resume_boundary_mismatch')
    del restored
    recovered_event=perform_step(model,opt,batches[1],cfg,'resume',2,3)
    recovered_model=clone_state(model.state_dict()); recovered_opt=clone_state(opt.state_dict())
    comparison={'next_loss_exact':main_events[1]['loss_before_update']==recovered_event['loss_before_update'],
                'next_loss_main':main_events[1]['loss_before_update'],
                'next_loss_resume':recovered_event['loss_before_update'],
                'model_difference':compare_state(main_model,recovered_model),
                'optimizer_difference':compare_state(main_opt,recovered_opt),
                'global_update_main':2,'global_update_resume':2,
                'frozen_parameter_changed_count':len(frozen_changes),
                'allowed_parameter_changed_count':len(allowed_changes),
                'allowed_parameter_changed_examples':allowed_changes[:10],
                'actual_optimizer_steps':3,'elapsed_seconds':time.monotonic()-started,
                'checkpoint_bytes':CHECKPOINT.stat().st_size,
                'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                'not_deployable':True,'eligibility':'infrastructure_smoke_only'}
    write_json(OUT/'smoke_resume_comparison.json',comparison)
    if not comparison['next_loss_exact'] or comparison['model_difference'] or comparison['optimizer_difference']:
        raise RuntimeError('resume_not_exact; see comparison')
    print(json.dumps(comparison))


if __name__=='__main__': main()
