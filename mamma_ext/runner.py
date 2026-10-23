"""Regular MAMMA plan, bounded train and ERP evaluation orchestration.

Importing this module never builds a model or reads a checkpoint. Current
configuration rejects execution; later runs require a separate authorized
configuration and explicit CLI arguments.
"""
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

from .policy import load_active_policy
from .split import group_inventory, validate_disjoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'configs/retrain/mamma_regular.json'


def _json_clean(value):
    if isinstance(value, dict):
        return {k:_json_clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_clean(value), ensure_ascii=False,
                                    indent=2, allow_nan=False)+'\n')


def read_plan(config_path, manifest_override=None):
    cfg, dataset = load_active_policy(config_path, manifest_override=manifest_override)
    return cfg, dataset, {
        'schema':'mamma_regular_plan_v1','configuration':str(Path(config_path).resolve()),
        'input_config':cfg['input_config'],'manifest':dataset.cfg['manifest'],
        'crop_policy_version':cfg['crop_policy_version'],
        'gt_policy':'saved_mamma_512_npz_v1',
        'target_grouping':group_inventory(dataset),
        'loss_enabled':cfg['loss_enabled'],'model_heads':cfg['model_heads'],
        'visibility_label_qualification':cfg['label_qualification'],
        'execution_authorized':cfg['execution_authorized'],
        'model_loaded':False,'image_reads':0}


def _execution_preflight(args, cfg, mode):
    if not args.execute:
        raise ValueError('explicit_execute_required')
    if cfg['execution_authorized'] is not True:
        raise ValueError('configuration_execution_not_authorized')
    if os.environ.get('PYTHON_DOTENV_DISABLED') != '1':
        raise ValueError('dotenv_must_be_disabled_for_execution')
    if not args.output or not args.device:
        raise ValueError('explicit_output_and_device_required')
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()) or output.exists():
        raise ValueError('output_must_be_new_directory_under_workspace_outputs')
    if not cfg.get('model_config') or not cfg.get('tensor_state'):
        raise ValueError('explicit_model_config_and_tensor_state_required')
    if mode == 'train':
        if not args.max_updates or args.max_updates <= 0 or not args.checkpoint_every or args.checkpoint_every <= 0:
            raise ValueError('positive_update_budget_and_checkpoint_interval_required')
        if cfg['loss_enabled']['visibility'] and cfg.get('label_qualification') != 'qualified':
            raise ValueError('visibility_training_labels_not_qualified')
    return output


def _tensor_batch(item, device):
    import torch
    keys = ('img','mask','xy_target','coordinate_loss_valid',
            'visibility','visibility_loss_valid')
    batch = {key:torch.as_tensor(np.ascontiguousarray(item[key][None]), device=device)
             for key in keys}
    for key in ('img','mask','xy_target'):
        batch[key] = batch[key].float()
    return batch


def _configure_numeric(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _load_original_model(cfg, device):
    from .model import deployment_config, strict_model_from_tensors, model_config_summary
    model_cfg = deployment_config(cfg['model_config'])
    summary = model_config_summary(model_cfg)
    if summary['num_joints'] != 512 or summary['image_size'] != [384,512] or not summary['normalize_plus_min_one']:
        raise ValueError('incompatible_mammanet_model_configuration')
    model, report = strict_model_from_tensors(model_cfg, cfg['tensor_state'])
    if model is None or not report['strict_load']:
        raise ValueError('strict_original_state_load_failed:'+repr(report))
    return model.to(device), report


def _refs(args, cfg, train_manifest, validation_manifest):
    return {'config':str(Path(args.config).resolve()),
            'train_manifest':str(Path(train_manifest).resolve()),
            'validation_manifest':str(Path(validation_manifest).resolve()),
            'model_config':str(Path(cfg['model_config']).resolve()),
            'tensor_state':str(Path(cfg['tensor_state']).resolve()),
            'crop_policy_version':cfg['crop_policy_version'],
            'gt_policy':'saved_mamma_512_npz_v1',
            'augmentation':cfg['augmentation']}


def visibility_metrics(labels, logits):
    """Do not turn a one-class observation into an apparent perfect score."""
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if labels.shape != logits.shape or not np.isin(labels, [0,1]).all():
        raise ValueError('invalid_visibility_metric_inputs')
    positive = int(np.count_nonzero(labels == 1))
    negative = int(np.count_nonzero(labels == 0))
    result = {'positive':positive,'negative':negative,'known':len(labels),
              'nonfinite_logit_count':int(np.count_nonzero(~np.isfinite(logits))),
              'status':'two_class_metric_available' if positive and negative else
                       'single_class_or_unlabeled_no_accuracy'}
    if positive and negative:
        if not np.isfinite(logits).all():
            result['status']='nonfinite_logits_no_accuracy'
            return result
        result['accuracy'] = float(np.mean((logits > 0) == labels))
        result['positive_recall'] = float(np.mean(logits[labels == 1] > 0))
        result['negative_recall'] = float(np.mean(logits[labels == 0] <= 0))
    return result


def train(args, cfg, dataset):
    output = _execution_preflight(args, cfg, 'train')
    if not len(dataset):
        raise ValueError('no_observed_training_targets')
    if not args.validation_manifest:
        raise ValueError('validation_manifest_required_for_grouped_split')
    _, validation = load_active_policy(args.config, manifest_override=args.validation_manifest)
    groups = validate_disjoint(dataset, validation)
    import torch
    from .policy import active_task_loss, configure_active_optimizer
    from .tasks import set_training_mode
    from .recovery import capture_rng, load_regular_boundary, save_regular_boundary
    _configure_numeric(cfg.get('seed',0))
    device = torch.device(args.device)
    model, strict_report = _load_original_model(cfg, device)
    optimizer, parameter_groups = configure_active_optimizer(model, cfg)
    refs = _refs(args,cfg,dataset.cfg['manifest'],validation.cfg['manifest'])
    update, epoch, position = 0,0,0
    if args.checkpoint:
        boundary = load_regular_boundary(args.checkpoint, model, optimizer, refs)
        update = boundary['global_update']
        epoch = boundary['sampler']['epoch']
        position = boundary['sampler']['next_position']
    if update >= args.max_updates:
        raise ValueError('resume_already_at_or_beyond_update_budget')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_plan.json', {'schema':'mamma_regular_run_v1',
        'status':'explicit_execution','max_updates':args.max_updates,
        'checkpoint_every':args.checkpoint_every,'strict_load':strict_report,
        'parameter_groups':parameter_groups,'groups':groups,'refs':refs})
    skipped = 0
    with (output/'updates.jsonl').open('w') as log:
        while update < args.max_updates:
            item = dataset[position]
            batch = _tensor_batch(item,device)
            set_training_mode(model)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch['img'],batch['mask'])
            result = active_task_loss(prediction,batch,cfg)
            next_position = position+1
            next_epoch = epoch
            if next_position == len(dataset):
                next_position = 0; next_epoch += 1
            if result['skip_optimizer']:
                skipped += 1
                if skipped >= len(dataset):
                    raise RuntimeError('entire_epoch_without_valid_update')
            else:
                skipped = 0
                if not torch.isfinite(result['loss']):
                    raise ValueError('nonfinite_training_loss')
                result['loss'].backward()
                optimizer.step()
                update += 1
            event = {'global_update':update,'epoch':epoch,'position':position,
                     'observation_id':item['identity']['observation_id'],
                     'loss':float(result['loss'].detach().cpu()),
                     'counts':result['counts'],'skip_optimizer':result['skip_optimizer'],
                     'skip_reason':result['skip_reason']}
            log.write(json.dumps(event,ensure_ascii=False)+'\n'); log.flush()
            position,epoch = next_position,next_epoch
            if update and not result['skip_optimizer'] and (
                update % args.checkpoint_every == 0 or update == args.max_updates):
                save_regular_boundary(output/f'checkpoint_update_{update:06d}.pt',
                    model, optimizer, update, capture_rng(),
                    {'epoch':epoch,'next_position':position},refs)
    write_json(output/'completed.json', {'global_update':update,'status':'complete',
                                       'not_deployable':True,'groups':groups})


def evaluate(args, cfg, dataset):
    output = _execution_preflight(args, cfg, 'evaluate')
    if not len(dataset):
        raise ValueError('no_observed_evaluation_targets')
    import torch
    from .model import predict_one, restore_xy, summarize_errors
    _configure_numeric(cfg.get('seed',0))
    device = torch.device(args.device)
    model, strict_report = _load_original_model(cfg,device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if payload.get('schema') != 'mamma_regular_resume_v2' or payload.get('not_deployable') is not True:
            raise ValueError('unsupported_evaluation_checkpoint')
        refs=payload.get('refs',{})
        if (refs.get('model_config') != str(Path(cfg['model_config']).resolve()) or
            refs.get('tensor_state') != str(Path(cfg['tensor_state']).resolve()) or
            refs.get('crop_policy_version') != cfg['crop_policy_version'] or
            refs.get('gt_policy') != dataset.cfg.get('gt_policy')):
            raise ValueError('evaluation_checkpoint_contract_mismatch')
        model.load_state_dict(payload['model_state'],strict=True)
    model.eval()
    output.mkdir(parents=True,exist_ok=False)
    rows=[]; all_errors=[]; vis_labels=[]; vis_logits=[]
    for index in range(len(dataset)):
        item = dataset[index]
        prediction,_ = predict_one(model,item,device)
        raw = prediction['joints2d'][0]
        model_xy, erp, affine_erp = restore_xy(raw[:,:2],item,True)
        if np.max(np.abs(erp-affine_erp)) > .01:
            raise ValueError('deployment_affine_restoration_disagree')
        errors,metrics = summarize_errors(erp,item['xy_erp'],item['coordinate_loss_valid'])
        all_errors.extend(errors[item['coordinate_loss_valid']].tolist())
        valid = item['visibility_loss_valid']
        if valid.any():
            vis_labels.extend(item['visibility'][valid].tolist())
            vis_logits.extend(prediction['visibility'][0,:,0][valid].tolist())
        np.savez_compressed(output/f'target_{index:05d}.npz',
            vertex_ids=item['vertex_ids'],raw_joints2d=raw,model_xy=model_xy,
            predicted_erp_xy=erp,gt_erp_xy=item['xy_erp'],
            coordinate_loss_valid=item['coordinate_loss_valid'],
            visibility=item['visibility'],visibility_loss_valid=valid,errors_erp_px=errors)
        rows.append({'identity':item['identity'],'metrics':metrics,
                     'prediction_file':f'target_{index:05d}.npz'})
    visibility = visibility_metrics(vis_labels,vis_logits)
    errors = np.asarray(all_errors)
    sample_means = [r['metrics']['mean_erp_px'] for r in rows
                    if r['metrics']['mean_erp_px'] is not None]
    write_json(output/'metrics.json', {'schema':'mamma_regular_eval_v1','rows':rows,
        'point_weighted_mean_erp_px':float('inf') if not np.isfinite(errors).all() else
                                     float(np.mean(errors)) if len(errors) else None,
        'sample_equal_mean_erp_px':float(np.mean(sample_means)) if sample_means else None,
        'sample_mean_denominator':len(sample_means),
        'visibility':visibility,'strict_load':strict_report,'crop_policy_version':cfg['crop_policy_version'],
        'gt_policy':'saved_mamma_512_npz_v1'})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','check-data','train','evaluate'):
        part=sub.add_parser(name)
        part.add_argument('--config',default=str(DEFAULT_CONFIG))
        part.add_argument('--manifest')
        if name == 'check-data':
            part.add_argument('--limit',type=int,default=2)
        if name in ('train','evaluate'):
            part.add_argument('--output',required=True)
            part.add_argument('--device',required=True)
            part.add_argument('--checkpoint')
            part.add_argument('--execute',action='store_true')
        if name == 'train':
            part.add_argument('--validation-manifest',required=True)
            part.add_argument('--max-updates',type=int,required=True)
            part.add_argument('--checkpoint-every',type=int,required=True)
    args=parser.parse_args(argv)
    if args.command in ('train','evaluate'):
        raw_cfg=json.loads(Path(args.config).read_text())
        _execution_preflight(args,raw_cfg,args.command)
    cfg,dataset,plan=read_plan(args.config,args.manifest)
    if args.command == 'plan':
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    if args.command == 'check-data':
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError('bounded_check_limit_required')
        rows=[]
        for index in range(args.limit):
            item=dataset[index]
            rows.append({'identity':item['identity'],'gt_source':item['gt_source'],
                         'image_shape':list(item['img'].shape),'mask_shape':list(item['mask'].shape),
                         'coordinate_valid':int(item['coordinate_loss_valid'].sum()),
                         'visibility_valid':int(item['visibility_loss_valid'].sum()),
                         'source_sampling_supported':item['source_sampling_supported']})
        result={'plan':plan,'checked_targets':rows,'model_loaded':False}
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    if args.command == 'train':
        return train(args,cfg,dataset)
    return evaluate(args,cfg,dataset)
