"""Historical original-weight box diagnostic; not active training input."""
import html
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'landmarks'))
import lib.models  # noqa: F401
from lib.baseline.mamma_original import deployment_config, predict_one, strict_model_from_tensors, restore_xy
from lib.datasets.mamma_roi_pilot import MammaManifestDataset
from lib.training.box_probe import VARIANTS, preflight, score_prediction

OUT=ROOT/'outputs/experiments/mamma_training_readiness_box_20261015'
CFG=json.loads((ROOT/'configs/retrain/mamma_box_probe_20261015.json').read_text())


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def panel(item,model_xy,variant,metrics):
    rgb=item['image_rgb_uint8'].copy()
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    invalid=~item['physical_valid_crop']
    bgr[invalid]=(bgr[invalid]*.68+np.array([165,80,35])*.32).astype(np.uint8)
    gt=item['xy_model']
    valid=item['coordinate_loss_valid']
    for k in range(0,512,8):
        for point,color in ((gt[k],(60,225,60) if valid[k] else (150,150,150)),
                            (model_xy[k],(210,40,210))):
            if np.isfinite(point).all() and 0<=point[0]<384 and 0<=point[1]<512:
                cv2.circle(bgr,tuple(np.rint(point).astype(int)),2,color,-1)
    label=np.zeros((62,384,3),np.uint8)
    text=[f'{variant} | common {metrics["common"]["valid_points"]} / own {metrics["own"]["valid_points"]}',
          f'common EPE {metrics["common"]["mean_erp_px"]:.1f}px | drift {metrics["common_drift_mean_erp_px"]:.1f}px',
          f'box {np.rint(item["parent_box_xyxy"]).astype(int).tolist()}']
    for i,line in enumerate(text):
        cv2.putText(label,line,(3,17+20*i),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1,cv2.LINE_AA)
    return cv2.resize(np.concatenate((bgr,label)),(192,287),interpolation=cv2.INTER_AREA)


def main():
    t=time.monotonic()
    if (OUT/'box_metrics.json').exists():
        raise RuntimeError('existing_box_diagnostic_refuses_overwrite')
    if os.environ.get('PYTHON_DOTENV_DISABLED')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('required_isolated_environment_missing')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('selected_gpu_unavailable')
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    d=MammaManifestDataset(CFG['manifest_reader'])
    rows,items,common=preflight(d)
    if len(rows)!=35:
        write_json(OUT/'box_support_runtime.json',rows)
        raise RuntimeError('planned_box_count_changed; see support table')
    if any((parent,'native') not in items for parent in range(5)):
        write_json(OUT/'box_support_runtime.json',rows)
        raise RuntimeError('native_parent_unsupported; see support table')
    write_json(OUT/'box_support_runtime.json',rows)
    base=json.loads(Path(CFG['baseline_config']).read_text())
    cfg=deployment_config(base['model_config'])
    model,load=strict_model_from_tensors(cfg,base['tensor_state'])
    write_json(OUT/'box_strict_load.json',load)
    if model is None or not load['strict_load']: raise RuntimeError('original_tensor_state_strict_load_failed')
    model.to('cuda:0').eval()
    forward_count=0
    native={}
    for parent in range(5):
        file=Path(base['output'])/f'slot_{parent}_prediction.npz'
        with np.load(file) as old:
            item=items[(parent,'native')]
            if not np.array_equal(old['coordinate_valid'],item['coordinate_loss_valid']) or not np.array_equal(old['affine_erp_to_model'],item['affine_erp_to_model']):
                raise RuntimeError('baseline_native_contract_mismatch')
            native[parent]=(old['raw_joints2d'].copy(),old['pred_erp_box_xy'].copy())
    check,info=predict_one(model,items[(0,'native')],torch.device('cuda:0'))
    forward_count+=1
    native_delta={k:float(np.max(np.abs(check[k][0]-native[0][0]))) for k in ('joints2d',)}
    if not np.array_equal(check['joints2d'][0],native[0][0]):
        write_json(OUT/'native_reuse_failure.json',native_delta)
        raise RuntimeError('slot0_native_prediction_not_identical')
    write_json(OUT/'native_reuse_check.json',{'max_abs_joints2d':native_delta['joints2d'],'forward_count':1,
                                             'input_dtype':info['input_dtype'],'output_shapes':info['output_shapes']})
    scored=[]; panels={i:[] for i in range(5)}
    for row in rows:
        if row['status']!='supported':
            continue
        if time.monotonic()-t>1100: raise TimeoutError('gpu_budget_near_limit')
        parent,name=row['parent'],row['variant']
        item=items[(parent,name)]
        if name=='native': raw=native[parent][0]
        else:
            output,info=predict_one(model,item,torch.device('cuda:0'))
            forward_count+=1
            raw=output['joints2d'][0]
        metrics,model_xy,erp=score_prediction(raw,item,common[parent],native[parent][1])
        entry={'parent':parent,'variant':name,'box_xyxy':row['box_xyxy'],
               'delta_center_fraction':row['delta_center_fraction'],'scale':row['scale'],
               'metrics':metrics,'prediction_file':f'box_slot{parent}_{name}.npz'}
        np.savez_compressed(OUT/entry['prediction_file'],vertex_ids=item['vertex_ids'],raw_joints2d=raw,
                            pred_model_xy=model_xy,pred_erp_xy=erp,gt_erp_xy=item['xy_erp'],
                            coordinate_valid=item['coordinate_loss_valid'],common_valid=common[parent],
                            parent_box_xyxy=item['parent_box_xyxy'],roi_xyxy=item['roi_xyxy'])
        scored.append(entry)
        panels[parent].append(panel(item,model_xy,name,metrics))
        write_json(OUT/'box_progress.json',{'forward_count':forward_count,'scored':len(scored),'seconds':time.monotonic()-t})
    for parent,images in panels.items():
        cv2.imwrite(str(OUT/f'box_slot{parent}_comparison.png'),np.concatenate(images,axis=1))
    mean_drift=[e['metrics']['common_drift_mean_erp_px'] for e in scored if e['variant']!='native']
    mean_epe=[e['metrics']['common']['mean_erp_px'] for e in scored]
    summary={'planned_boxes':35,'supported_boxes':len(items),'rejected_boxes':35-len(items),'actual_forward_count':forward_count,
             'native_reused':5,'native_verified_slot':0,'common_valid_points':[int(common[i].sum()) for i in range(5)],
             'mean_common_epe_across_boxes_erp_px':float(np.mean(mean_epe)),
             'mean_drift_across_variants_erp_px':float(np.mean(mean_drift)),
             'max_variant_drift_mean_erp_px':float(np.max(mean_drift)),
             'seconds':time.monotonic()-t,'device_uuid':'GPU-d481acc7-106f-29e5-1298-8c0d398261b2'}
    write_json(OUT/'box_metrics.json',{'summary':summary,'boxes':scored})
    cards=''.join(f'<section><h2>slot {i}</h2><p>共同有效点 {int(common[i].sum())}/512；紫色预测，绿色有效 GT，灰色无效 GT；每 8 点展示 1 点，评价用所有点。蓝色阴影是物理域外；各图为模型 crop。</p><img src="box_slot{i}_comparison.png"></section>' for i in range(5))
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>MAMMA 同图换框</title><style>body{background:#17191e;color:#eee;font:16px system-ui;margin:25px}section{margin:20px 0}img{max-width:100%;height:auto}a{color:#9cf}</style><h1>五样本同图换框诊断</h1><p>原权重；oracle mask；七种固定框。ROI 图像不重建全景。指标使用每点完整 ERP 坐标，第三值不筛点。<a href="box_metrics.json">指标 JSON</a></p>'+cards+'</html>')
    print(json.dumps(summary))


if __name__=='__main__': main()
