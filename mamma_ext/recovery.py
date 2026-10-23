"""Full deterministic update-boundary state for bounded MAMMA training."""
import copy
import random

import numpy as np
import torch


def capture_rng():
    n=np.random.get_state()
    return {'python':random.getstate(), 'numpy':{'name':n[0], 'keys':n[1].tolist(),
            'pos':n[2], 'has_gauss':n[3], 'cached_gaussian':n[4]},
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(s):
    random.setstate(s['python'])
    n=s['numpy']
    np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(s['torch_cpu'])
    if s['torch_cuda']:
        if not torch.cuda.is_initialized():
            raise ValueError('checkpoint_cuda_rng_requires_initialized_cuda')
        torch.cuda.set_rng_state_all(s['torch_cuda'])


def clone_state(state):
    if torch.is_tensor(state): return state.detach().cpu().clone()
    if isinstance(state,dict): return {k:clone_state(v) for k,v in state.items()}
    if isinstance(state,list): return [clone_state(v) for v in state]
    if isinstance(state,tuple): return tuple(clone_state(v) for v in state)
    return copy.deepcopy(state)


def compare_state(a,b,path=''):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape!=b.shape: return {'path':path,'reason':'tensor_shape_or_type'}
        if not torch.equal(a.cpu(),b.cpu()):
            delta=(a.cpu().float()-b.cpu().float()).abs()
            return {'path':path,'reason':'tensor_values','max_abs':float(delta.max()),
                    'unequal':int(torch.count_nonzero(a.cpu()!=b.cpu()))}
        return None
    if type(a)!=type(b): return {'path':path,'reason':'type'}
    if isinstance(a,dict):
        if a.keys()!=b.keys(): return {'path':path,'reason':'keys'}
        for k in a:
            failure=compare_state(a[k],b[k],path+'/'+str(k))
            if failure: return failure
        return None
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b): return {'path':path,'reason':'length'}
        for k,(x,y) in enumerate(zip(a,b)):
            failure=compare_state(x,y,path+'/'+str(k))
            if failure: return failure
        return None
    return None if a==b else {'path':path,'reason':'scalar','a':str(a),'b':str(b)}


def save_update_boundary(path, model, optimizer, global_update, rng, sampler, box_stream, refs):
    """Write historical v1 continuation state; box_stream is legacy off-only metadata."""
    if box_stream.get('augmentation') != 'off':
        raise ValueError('legacy_box_stream_requires_augmentation_off')
    payload={'schema':'mamma_infrastructure_resume_v1','eligibility':'infrastructure_smoke_only',
             'not_deployable':True,'model_state':model.state_dict(),
             'optimizer_state':optimizer.state_dict(),'global_update':global_update,
             'rng':rng,'sampler':sampler,'box_stream':box_stream,'refs':refs}
    torch.save(payload,path)


def load_update_boundary(path, model, optimizer):
    """Load v1, including its legacy box_stream, without changing old schema."""
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!='mamma_infrastructure_resume_v1' or payload.get('not_deployable') is not True:
        raise ValueError('invalid_resume_schema')
    model.load_state_dict(payload['model_state'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    restore_rng(payload['rng'])
    return {k:v for k,v in payload.items() if k not in ('model_state','optimizer_state','rng')}


def regular_payload(model_state, optimizer_state, global_update, rng, sampler, refs):
    """New normal-run boundary, distinct from the historical smoke v1 schema."""
    required = ('config','train_manifest','validation_manifest','model_config',
                'tensor_state','crop_policy_version','gt_policy','augmentation')
    if not isinstance(global_update,int) or global_update < 0 or any(not refs.get(k) for k in required):
        raise ValueError('incomplete_regular_checkpoint_boundary')
    if not isinstance(sampler,dict) or not {'epoch','next_position'} <= sampler.keys():
        raise ValueError('incomplete_regular_sampler_state')
    return {'schema':'mamma_regular_resume_v2','not_deployable':True,
            'model_state':model_state,'optimizer_state':optimizer_state,
            'global_update':global_update,'rng':rng,'sampler':sampler,'refs':refs}


def save_regular_boundary(path, model, optimizer, global_update, rng, sampler, refs):
    payload = regular_payload(model.state_dict(), optimizer.state_dict(),
                              global_update, rng, sampler, refs)
    torch.save(payload, path)


def load_regular_boundary(path, model, optimizer, expected_refs):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != 'mamma_regular_resume_v2' or
        payload.get('not_deployable') is not True or payload.get('refs') != expected_refs):
        raise ValueError('regular_resume_schema_or_refs_mismatch')
    model.load_state_dict(payload['model_state'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    restore_rng(payload['rng'])
    return {'global_update':payload['global_update'],'sampler':payload['sampler'],
            'refs':payload['refs']}
