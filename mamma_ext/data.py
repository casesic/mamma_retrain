"""Current MAMMA bundle reader: RGB, target mask and saved ordered 512 labels."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .upstream import DEFAULT_MEAN,DEFAULT_STD,crop_math

expand_to_aspect_ratio,gen_trans_from_patch_cv=crop_math()


def mask_box_inclusive(mask,offset):
    if mask.ndim!=2 or mask.dtype!=np.uint8 or not np.isin(mask,[0,255]).all():
        raise ValueError('invalid_binary_person_mask')
    x,y,w,h=cv2.boundingRect(mask)
    if w<2 or h<2:raise ValueError('empty_or_degenerate_person_mask')
    return np.array([offset[0]+x,offset[1]+y,offset[0]+x+w-1,offset[1]+y+h-1],np.int64)


def native_plan(box,roi,stored,full):
    box=np.asarray(box,np.float32);roi=np.asarray(roi,np.int64)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
        raise ValueError('invalid_mask_box')
    if roi.shape!=(4,) or list(roi[2:]-roi[:2])!=list(stored) or np.any(roi[:2]<0) or np.any(roi[2:]>full):
        raise ValueError('invalid_roi')
    center=(box[:2]+box[2:])/2
    size=expand_to_aspect_ratio((box[2:]-box[:2])*1.2,(384,512))
    downsampling=size.max()/384/2
    sigma=(downsampling-1)/2 if downsampling>1.1 else 0
    margin=int(np.ceil(4*sigma))+3
    rect=np.r_[np.floor(center-size/2).astype(int)-margin,np.ceil(center+size/2).astype(int)+margin]
    if np.any(rect[:2]<roi[:2]) or np.any(rect[2:]>roi[2:]):
        raise ValueError('crop_filter_support_outside_roi')
    return center,size,sigma,rect


class MammaSavedGTDataset(torch.utils.data.Dataset):
    def __init__(self,config_path,manifest_override=None):
        config_path=Path(config_path).resolve()
        cfg=json.loads(config_path.read_text())
        if set(cfg)!={'schema','manifest'} or cfg['schema']!='mamma_reader_v1':
            raise ValueError('current_reader_config_required')
        manifest=Path(manifest_override or cfg['manifest'])
        if not manifest.is_absolute():manifest=config_path.parent/manifest
        manifest=manifest.resolve()
        bundle=json.loads(manifest.read_text())
        if bundle.get('schema')!='mamma_bundle_v1':
            raise ValueError('current_bundle_required')
        if bundle.get('template_id')!='MAMMA_verts_512_ordered_v1':
            raise ValueError('unsupported_template_id')
        if (not bundle['frames'] or bundle['frames']!=sorted(set(bundle['frames'])) or
            any(type(k) is not int or not 0<=k<bundle['frame_count'] for k in bundle['frames']) or
            len(bundle['rig_ids'])!=len(set(bundle['rig_ids'])) or
            len(bundle['camera_ids'])!=len(set(bundle['camera_ids']))):
            raise ValueError('invalid_bundle_axes')
        self.cfg=dict(cfg,manifest=str(manifest))
        self.slots=bundle['observations']
        self.ready=[]
        self._files=[]
        expected=set()
        template=np.asarray(bundle['ordered_vertex_ids'])
        if (template.shape!=(512,) or len(np.unique(template))!=512 or
            np.any(template<0) or np.any(template>=10475)):
            raise ValueError('invalid_template_point_order')
        for index,row in enumerate(self.slots):
            key=(row['episode'],row['rig'],row['frame_index'],row['camera_id'],row['person_id'])
            if key in expected:raise ValueError('duplicate_observation')
            expected.add(key)
            if (row['episode']!=bundle['episode'] or row['rig'] not in bundle['rig_ids'] or
                row['frame_index'] not in bundle['frames'] or row['camera_id'] not in bundle['camera_ids'] or
                row['status'] not in ('observed','absent') or
                (row['status']=='absent')!=(row['absent_reason'] is not None)):
                raise ValueError('bundle_identity_or_status_conflict')
            files={}
            for name in ('rgb','mask','labels','annotations'):
                path=(manifest.parent/row[name]).resolve()
                if not path.is_relative_to(manifest.parent) or not path.is_file():
                    raise ValueError('missing_or_external_bundle_file:'+name)
                files[name]=path
            self._files.append(files)
            with np.load(files['labels'],allow_pickle=False) as gt,np.load(files['annotations'],allow_pickle=False) as ann:
                if any(name in gt for name in
                       ('visibility_known','contact_value','contact_known',
                        'floor_contact_value','floor_contact_known','sidecar_policy')):
                    raise ValueError('obsolete_label_fields')
                pids=gt['person_ids'].astype(int).tolist();j=row['person_index']
                if (str(gt['schema'])!='mamma_512_label_v1' or
                    j<0 or j>=len(pids) or pids[j]!=row['person_id'] or
                    not np.array_equal(gt['person_ids'],ann['person_ids']) or
                    int(gt['frame_index'])!=row['frame_index'] or int(ann['frame_index'])!=row['frame_index'] or
                    str(gt['camera_id'])!=row['camera_id'] or str(gt['rig_id'])!=row['rig'] or
                    str(gt['scene_episode_id'])!=row['episode'] or str(ann['scene_episode_id'])!=row['episode'] or
                    str(ann['rig_id'])!=row['rig'] or int(gt['time_tick'])!=row['time_tick'] or
                    int(ann['time_tick'])!=row['time_tick'] or
                    gt['time_base'].tolist()!=row['time_base'] or ann['time_base'].tolist()!=row['time_base'] or
                    int(gt['asset_frame_index'][j])!=row['asset_frame_index'] or
                    int(ann['person_asset_frame'][j])!=row['asset_frame_index'] or
                    int(gt['source_frame_index'][j])!=row['source_frame_index'] or
                    int(ann['person_source_frame'][j])!=row['source_frame_index'] or
                    str(gt['template_id'])!=bundle['template_id'] or
                    not np.array_equal(gt['ordered_vertex_ids'],template) or
                    not np.array_equal(gt['render_coverage_xyxy'],row['roi_xyxy']) or
                    not np.array_equal(gt['full_image_size_wh'],row['full_image_size_wh']) or
                    ann['camera_to_world_erp'].shape!=(4,4) or
                    not np.isfinite(ann['camera_to_world_erp']).all() or
                    gt['xy_full_erp'].shape!=(len(pids),512,2) or
                    gt['visibility_value'].shape!=(len(pids),512) or
                    not np.isfinite(gt['xy_full_erp']).all() or
                    not np.isin(gt['visibility_value'],[-1,0,1]).all()):
                    raise ValueError('saved_gt_identity_time_or_template_conflict')
            if row['status']=='observed':
                if row['crop_support'] is None or row['crop_support']['crop_policy_version']!='mask_bbox_inclusive_v1':
                    raise ValueError('missing_mask_crop_support')
                self.ready.append(index)
            elif row['crop_support'] is not None:
                raise ValueError('absent_slot_has_crop')
        self.vertex_ids=template

    def __len__(self):
        return len(self.ready)

    def slot_status(self,index):
        row=self.slots[index]
        return {key:row[key] for key in ('episode','rig','observation_id','status','person_id')} | {
            'reason':row['absent_reason']}

    def observation_index(self,episode,rig,frame_index,camera_id,person_id):
        matches=[i for i,row in enumerate(self.slots) if
                 (row['episode'],row['rig'],row['frame_index'],row['camera_id'],row['person_id'])==
                 (episode,rig,frame_index,camera_id,person_id)]
        if len(matches)!=1:raise ValueError('observation_not_unique_or_missing')
        if matches[0] not in self.ready:raise ValueError('observation_absent')
        return self.ready.index(matches[0])

    def __getitem__(self,index):
        slot=self.ready[index];row=self.slots[slot];files=self._files[slot]
        image=cv2.imread(str(files['rgb']),cv2.IMREAD_COLOR)
        mask=cv2.imread(str(files['mask']),cv2.IMREAD_GRAYSCALE)
        roi=np.asarray(row['roi_xyxy'],np.int64);full=np.asarray(row['full_image_size_wh'],np.int64)
        stored=roi[2:]-roi[:2]
        if image is None or mask is None or image.shape[:2]!=(stored[1],stored[0]) or mask.shape!=image.shape[:2]:
            raise ValueError('stored_raster_size_conflict')
        box=mask_box_inclusive(mask,roi[:2])
        center,size,sigma,rect=native_plan(box,roi,stored,full)
        support=row['crop_support']
        if support['box_xyxy']!=box.tolist() or support['sampling_rect_erp']!=rect.tolist() or not support['source_sampling_supported']:
            raise ValueError('crop_support_disagrees_with_mask')
        x0,y0,x1,y1=rect-np.tile(roi[:2],2)
        local=image[y0:y1,x0:x1]
        local_mask=mask[y0:y1,x0:x1]
        if sigma:
            local=cv2.GaussianBlur(local.astype(np.float32),(0,0),sigmaX=sigma,sigmaY=sigma,
                                   borderType=cv2.BORDER_REFLECT)
        affine=gen_trans_from_patch_cv(center[0],center[1],size[0],size[1],384,512,1.0,0)
        local_affine=affine.copy()
        local_affine[:,2]+=affine[:,:2]@rect[:2]
        patch=cv2.warpAffine(local,local_affine,(384,512),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        mask_patch=cv2.warpAffine(local_mask,local_affine,(384,512),flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT)
        rgb=patch[:,:,::-1].transpose(2,0,1).astype(np.float32)
        for channel in range(3):
            rgb[channel]=(rgb[channel]-DEFAULT_MEAN[channel])/DEFAULT_STD[channel]
        with np.load(files['labels'],allow_pickle=False) as gt:
            j=row['person_index']
            xy=gt['xy_full_erp'][j].astype(np.float64)
            geometry=gt['geometry_defined'][j].astype(bool)
            physical=gt['physical_domain'][j].astype(bool)
            coverage=gt['render_coverage'][j].astype(bool)
            visibility=gt['visibility_value'][j].astype(np.int8)
        model_xy=xy@affine[:,:2].T+affine[:,2]
        in_crop=np.all((model_xy>=0)&(model_xy<[384,512]),axis=1)
        in_roi=np.all((xy>=roi[:2])&(xy<roi[2:]),axis=1)
        coordinate_valid=geometry&physical&coverage&in_roi&in_crop&np.isfinite(xy).all(1)
        return dict(img=rgb,mask=mask_patch[None].astype(np.float32)/255,
                    xy_erp=xy,xy_model=model_xy,xy_target=model_xy/np.array([384,512])*2-1,
                    coordinate_loss_valid=coordinate_valid,
                    visibility=visibility,visibility_loss_valid=(visibility!=-1)&coordinate_valid,
                    vertex_ids=self.vertex_ids.copy(),identity=self.slot_status(slot),
                    affine_erp_to_model=affine,affine_model_to_erp=cv2.invertAffineTransform(affine),
                    roi_xyxy=roi,full_size_wh=full,source_sampling_supported=True,
                    gt_source='saved_mamma_512_npz_v1')
