"""One trusted local CPU safe-load; never constructs a model."""
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acceptance/mamma_data_adapter_20261013/checkpoint_metadata.json'
SOURCE=Path('/home/yy/Repos/mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt')
t0=time.monotonic()
report=dict(source=str(SOURCE),python=sys.executable,torch=torch.__version__,
            load_policy='weights_only=True,map_location=cpu',
            no_model_constructed=True,strict_load_performed=False)
try:
    c=torch.load(SOURCE,map_location='cpu',weights_only=True)
    report['top_level_keys']=list(c) if isinstance(c,dict) else None
    if isinstance(c,dict):
        report['epoch']=c.get('epoch')
        report['global_step']=c.get('global_step')
        report['pytorch_lightning_version']=c.get('pytorch-lightning_version')
        state=c.get('state_dict')
        if isinstance(state,dict):
            report['state_key_count']=len(state)
            report['state_key_prefixes']=sorted({k.split('.')[0] for k in state})
            report['selected_shapes']={k:list(v.shape) for k,v in state.items() if
                ('query_embed' in k or 'landmarks.layers' in k or 'vis_prob' in k or
                 'contact_prob' in k or 'mask_downscaling' in k)}
            report['shape_counts']={}
            for v in state.values():
                shape=str(tuple(v.shape))
                report['shape_counts'][shape]=report['shape_counts'].get(shape,0)+1
        h=c.get('hyper_parameters')
        report['hyper_parameters_type']=type(h).__name__ if h is not None else None
        if isinstance(h,dict):
            report['hyper_parameter_keys']=list(h)
            for key in ('model','model_cfg','config','cfg'):
                if key in h:
                    report['model_config_repr']=repr(h[key])[:4000]
                    break
        report['status']='safe_metadata_read'
    else:report['status']='unexpected_checkpoint_type'
except Exception as e:
    report['status']='safe_read_unavailable'
    report['error']=type(e).__name__+': '+str(e)[:1200]
report['elapsed_seconds']=time.monotonic()-t0
OUT.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('selected_shapes','shape_counts','model_config_repr')},indent=2,default=str))
