"""Fixed five-slot original MammaNet PyTorch baseline; six total forward calls."""
import gc
import html
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'landmarks'))
import lib.models  # Upstream datasets/models package import cycle requires this order.
from lib.baseline.mamma_original import (deployment_config, model_config_summary,
                                         predict_one, restore_xy, strict_model_from_tensors,
                                         summarize_errors)
from lib.datasets.mamma_roi_pilot import MammaROIPilot

CONFIG = ROOT / 'configs/retrain/mamma_original_baseline_pilot_20261015.json'


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v): return 'Infinity' if v > 0 else '-Infinity' if v < 0 else 'NaN'
        return v
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def overlay(item, model_xy, valid, slot, output_path):
    rgb8 = np.clip(item['image_rgb_uint8'], 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    invalid = ~item['coverage_crop']
    if invalid.any():
        tint = np.empty_like(bgr)
        tint[:] = (165, 80, 35)
        bgr[invalid] = (0.68 * bgr[invalid] + 0.32 * tint[invalid]).astype(np.uint8)
    gt = item['xy_model']
    for point_id in range(0, 512, 4):
        gx, gy = gt[point_id]
        px, py = model_xy[point_id]
        if np.isfinite([gx, gy]).all() and 0 <= gx < 384 and 0 <= gy < 512:
            cv2.circle(bgr, (round(gx), round(gy)), 2, (60, 225, 60) if valid[point_id] else (160, 160, 160), -1)
        if valid[point_id] and np.isfinite([px, py]).all() and 0 <= px < 384 and 0 <= py < 512:
            cv2.circle(bgr, (round(px), round(py)), 2, (210, 40, 210), -1)
    for point_id in (0, 64, 128, 192, 256, 320, 384, 448):
        x, y = gt[point_id]
        if np.isfinite([x, y]).all() and 0 <= x < 384 and 0 <= y < 512:
            cv2.putText(bgr, str(point_id), (min(360, round(x)+3), max(12, round(y)-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, str(point_id), (min(360, round(x)+3), max(12, round(y)-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.zeros((98, 384, 3), np.uint8)
    lines = [f"{slot['episode']} {slot['rig']} {slot['observation_id']}",
             f"ROI offset {item['roi_offset_xy'].tolist()} | valid {int(valid.sum())}/512",
             'Green GT  Magenta pred  Gray GT invalid',
             'Blue tint: physical invalid; visibility GT UNKNOWN',
             'Display every 4th ID; metrics use all valid IDs']
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (6, 17 + i*19), cv2.FONT_HERSHEY_SIMPLEX, .38,
                    (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.concatenate([bgr, panel], axis=0))


def index_html(out, rows, summary):
    cards=[]
    for r in rows:
        title=html.escape(f"{r['episode']} / {r['rig']} / {r['observation_id']} / p{r['person_id']}")
        cards.append(f"<section><h2>{title}</h2><p>有效点 {r['metrics']['valid_points']}/512；ERP 均值 {r['metrics']['mean_erp_px']:.2f} px，中位数 {r['metrics']['median_erp_px']:.2f} px，P90 {r['metrics']['p90_erp_px']:.2f} px。</p><img src=\"{r['overlay']}\" alt=\"GT 与原权重预测叠图\"><p><a href=\"{r['prediction_file']}\">逐点预测 NPZ</a></p></section>")
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MAMMA 原权重五样本基线</title><style>body{font:16px system-ui;max-width:980px;margin:30px auto;background:#17191e;color:#eee}section{display:inline-block;vertical-align:top;width:450px;margin:12px;padding:16px;background:#252a33;border-radius:8px}img{width:384px}a{color:#9ccfff}p{line-height:1.5}</style><h1>MAMMA 原权重五样本 PyTorch 基线</h1><p>integration_only；原生 oracle 框与 mask。可见性 GT 未知，展示每四个编号中的一个；指标使用全部坐标有效点。蓝色区域来自校准 physical mask，图片只显示 384×512 模型输入。</p>'''
    page += f"<p>点加权 ERP 均值 {summary['point_weighted_mean_erp_px']:.2f} px；样本均值 {summary['sample_mean_erp_px']:.2f} px。</p>"
    page += ''.join(cards) + '</html>'
    (out/'index.html').write_text(page)


def main():
    started=time.monotonic()
    cfg_run=json.loads(CONFIG.read_text())
    out=Path(cfg_run['output'])
    if (out/'metrics.json').exists(): raise RuntimeError('existing_finished_baseline_refuse_overwrite')
    if os.environ.get('PYTHON_DOTENV_DISABLED') != '1': raise RuntimeError('dotenv_must_be_disabled')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg_run['cuda_visible_devices']: raise RuntimeError('device_visibility_mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError('selected_single_cuda_unavailable')
    torch.manual_seed(cfg_run['seed']); np.random.seed(cfg_run['seed']); torch.cuda.manual_seed_all(cfg_run['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    config=deployment_config(cfg_run['model_config'])
    config_summary=model_config_summary(config)
    if config_summary['image_size'] != [384,512] or config_summary['num_joints'] != 512 or not config_summary['normalize_plus_min_one']:
        raise RuntimeError('model_input_contract_mismatch')
    model, load_report=strict_model_from_tensors(config,cfg_run['tensor_state'])
    write_json(out/'strict_load.json',load_report)
    if model is None or not load_report['strict_load'] or not load_report['backbone_keys'] == load_report['checkpoint_backbone_keys']:
        raise RuntimeError('strict_full_model_load_failed')
    model.to(device).eval()
    d=MammaROIPilot(cfg_run['input_config'])
    if len(d)!=5 or len(d.slots)!=6 or d.slots[3]['reason']!='outside_physical_domain_no_instance_box':
        raise RuntimeError('fixed_slots_changed')
    if not str(Path(sys.modules['lib.models'].__file__).resolve()).startswith(str(ROOT)) or not str(Path(sys.modules['lib.datasets.mamma_roi_pilot'].__file__).resolve()).startswith(str(ROOT)):
        raise RuntimeError('new_workspace_import_mismatch')
    rows=[]; all_errors=[]; repeat_item=None; repeat_output=None
    for i in range(5):
        if time.monotonic()-started > 1100: raise TimeoutError('numerical_budget_near_limit')
        item=d[i]; slot=d.slots[d.ready[i]]
        if repeat_item is None: repeat_item=item
        pred,info=predict_one(model,item,device)
        expected={'joints2d':[1,512,3], 'visibility':[1,512,1],
                  'contact':[1,512,1], 'floor_contact':[1,512,1]}
        if info['output_shapes']!=expected: raise RuntimeError('model_output_shape_mismatch:'+repr(info['output_shapes']))
        if i==cfg_run['repeat_slot']: repeat_output=pred
        raw=pred['joints2d'][0]
        model_xy,erp_box,erp_affine=restore_xy(raw[:,:2],item,True)
        affine_difference=np.linalg.norm(erp_box-erp_affine,axis=1)
        if not np.isfinite(affine_difference).all() or affine_difference.max()>0.01:
            write_json(out/f'slot_{i}_restore_failure.json',dict(max_difference_px=float(np.nanmax(affine_difference)), sample=i))
            raise RuntimeError('deployment_vs_affine_restore_mismatch')
        errors,metrics=summarize_errors(erp_box,item['xy_erp'],item['coordinate_loss_valid'])
        valid=item['coordinate_loss_valid']
        all_errors.extend(errors[valid].tolist())
        metric=dict(slot=i,episode=slot['episode'],rig=slot['rig'],observation_id=slot['observation_id'],
                    person_id=slot['person_id'],source_frame=item['source_frame'],
                    source_time_tick=item['source_time_tick'],asset_frame_index=item['asset_frame_index'],
                    roi_offset_xy=item['roi_offset_xy'].tolist(),parent_box_xyxy=item['parent_box_xyxy'].tolist(),
                    full_size_wh=item['full_size_wh'].tolist(),input_dtypes=info['input_dtype'],
                    output_shapes=info['output_shapes'],output_dtypes=info['output_dtypes'],
                    output_nonfinite_counts=info['nonfinite_counts'], metrics=metrics,
                    restore_box_vs_affine_max_px=float(affine_difference.max()),
                    restore_box_vs_affine_mean_px=float(affine_difference.mean()),
                    prediction_file=f'slot_{i}_prediction.npz',overlay=f'slot_{i}_overlay.png',
                    visibility_gt='unknown',contact_gt='unknown',floor_contact_gt='unknown')
        np.savez_compressed(out/metric['prediction_file'],
                            vertex_ids=item['vertex_ids'],raw_joints2d=raw,
                            raw_visibility_logit=pred['visibility'][0],
                            raw_contact_logit=pred['contact'][0],
                            raw_floor_contact_logit=pred['floor_contact'][0],
                            pred_model_xy=model_xy,pred_erp_box_xy=erp_box,
                            pred_erp_affine_xy=erp_affine,gt_erp_xy=item['xy_erp'],
                            gt_model_xy=item['xy_model'],coordinate_valid=valid,
                            euclidean_error_erp_px=errors,physical_domain=item['physical_domain'],
                            roi_coverage=item['roi_coverage'],crop_support=item['crop_support'],
                            affine_erp_to_model=item['affine_erp_to_model'],
                            affine_model_to_erp=item['affine_model_to_erp'],
                            raw_third=raw[:,2],source_frame=np.int64(item['source_frame']),
                            source_time_tick=np.int64(item['source_time_tick']))
        overlay(item,model_xy,valid,slot,out/metric['overlay'])
        rows.append(metric)
        write_json(out/'progress_results.json',dict(status='five_prediction_stage_running', completed=rows,
                                                     forward_count=len(rows),elapsed_seconds=time.monotonic()-started))
        print('predicted',i,slot['observation_id'],'valid',metrics['valid_points'],'mean',metrics['mean_erp_px'],flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    if time.monotonic()-started > 1100: raise TimeoutError('numerical_budget_near_limit_before_reload')
    torch.manual_seed(cfg_run['seed']); torch.cuda.manual_seed_all(cfg_run['seed'])
    model2, reload_report=strict_model_from_tensors(config,cfg_run['tensor_state'])
    write_json(out/'reload_strict.json',reload_report)
    if model2 is None or not reload_report['strict_load']: raise RuntimeError('reload_strict_failed')
    model2.to(device).eval()
    repeat_pred,repeat_info=predict_one(model2,repeat_item,device)
    repeat_delta={k:dict(max_abs=float(np.max(np.abs(repeat_pred[k]-repeat_output[k]))),
                         mean_abs=float(np.mean(np.abs(repeat_pred[k]-repeat_output[k]))),
                         unequal_elements=int(np.count_nonzero(repeat_pred[k]!=repeat_output[k])))
                  for k in repeat_output}
    write_json(out/'reload_comparison.json',dict(slot=cfg_run['repeat_slot'],
                  output_dtypes=repeat_info['output_dtypes'],per_head=repeat_delta,
                  same_config_and_tensor_state=True,model_reconstructed=True,
                  forward_count_total=6))
    paired=sorted(set(np.flatnonzero(d[0]['coordinate_loss_valid'])) & set(np.flatnonzero(d[1]['coordinate_loss_valid'])))
    with np.load(out/'slot_0_prediction.npz') as a, np.load(out/'slot_1_prediction.npz') as b:
        ea=a['euclidean_error_erp_px'][paired]; eb=b['euclidean_error_erp_px'][paired]
        pair=dict(common_valid_vertex_count=len(paired),common_vertex_ids=d.vertex_ids[paired].tolist(),
                  cam2_mean_own_gt_residual_px=float(np.mean(ea)),
                  cam4_mean_own_gt_residual_px=float(np.mean(eb)),
                  pair_mean_of_two_own_gt_residuals_px=float(np.mean((ea+eb)/2)),
                  definition='same 512 query IDs valid in both e01 cameras; average each camera own ERP pixel distance to its own GT; no camera pixel subtraction')
    if not all(np.isfinite(all_errors)):
        point_mean=float('inf')
    else:point_mean=float(np.mean(all_errors))
    sample_mean=float(np.mean([r['metrics']['mean_erp_px'] for r in rows]))
    summary=dict(schema='mamma_original_baseline_pilot_v1',status='completed',
                 model_config=config_summary,checkpoint_state=str(cfg_run['tensor_state']),
                 strict_load=load_report,reload_strict=reload_report,
                 point_weighted_mean_erp_px=point_mean,sample_mean_erp_px=sample_mean,
                 evaluated_point_count=sum(r['metrics']['valid_points'] for r in rows),
                 sample_count=5,forward_count=6,no_box_slot=d.slot_status(3),
                 paired_e01=pair,rows=rows,repeat=repeat_delta,
                 device=dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                             configured_gpu_uuid=cfg_run['device_uuid'],
                             name=torch.cuda.get_device_name(0),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(0)),
                 python=sys.executable,torch=torch.__version__,
                 imports=dict(model=sys.modules['lib.models'].__file__,
                              adapter=sys.modules['lib.datasets.mamma_roi_pilot'].__file__,
                              baseline=sys.modules['lib.baseline.mamma_original'].__file__),
                 precision='float32',amp=False,cudnn_benchmark=False,tf32=False,
                 seed=cfg_run['seed'],elapsed_seconds=time.monotonic()-started,
                 integration_only=True)
    write_json(out/'metrics.json',summary)
    index_html(out,rows,summary)
    print('completed',summary['elapsed_seconds'],'point_mean',point_mean,'sample_mean',sample_mean,flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        out=ROOT/'outputs/experiments/mamma_original_baseline_pilot_20261015'
        write_json(out/'inference_failure.json',dict(status='failed',type=type(exc).__name__,error=str(exc)))
        raise
