import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from misc.utils import get_local_rank, get_local_size
from torch.utils.data import Dataset
import json
import einops
class RGBTDataSet(Dataset):
    def __init__(self,root_rgb,root_tir,annFile,transforms,max_len=5000,local_rank=0,local_size=1):
        super().__init__()
        self.transforms=transforms
        self.max_len=max_len
        self.local_rank=local_rank
        self.local_size=local_size
        self.root_rgb=root_rgb
        self.root_tir=root_tir
        with open(annFile,'r') as f:
            self.anns=json.load(f)
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self,idx):
        if idx>=self.max_len:
            return None
        if idx%self.local_size!=self.local_rank:
            return None
        ann=self.anns[idx]
        rgb_path=os.path.join(self.root_rgb,ann['rgb_file_name'])
        tir_path=os.path.join(self.root_tir,ann['tir_file_name'])
        rgb=cv2.imread(rgb_path)
        h,w=rgb.shape[:2]
        tir=cv2.imread(tir_path)
        if rgb is None or tir is None:
            return None
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        tir=cv2.cvtColor(tir,cv2.COLOR_BGR2RGB)
        keypoints=ann["points"]
        keypoints_in_image=[]
        for pt in keypoints:
            if pt[0]>=w or pt[1]>=h or pt[0]<0 or pt[1]<0:
                continue
            else:
                keypoints_in_image.append(pt)
        keypoints=keypoints_in_image
        if self.transforms is not None:
            data=self.transforms(image=rgb,tir=tir,keypoints=keypoints)
            rgb=data['image']
            tir=data['tir']
            tir=einops.reduce(tir,"(c1 c) h w->c1 h w","mean",c1=1)
            keypoints=data['keypoints']

        labels={}
        labels["num"]=torch.tensor(len(keypoints))
        labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        if labels["num"] > 0:
            labels["points"][:len(keypoints)] = torch.tensor(keypoints)[:self.max_len]
        if labels["num"] > self.max_len:
            print(
                f"Warning: the number of points {labels['num']} is larger than max_len {self.max_len}")
            labels["num"] = torch.as_tensor(self.max_len, dtype=torch.long)
        image=torch.cat([rgb,tir],dim=0)
        return image,labels


def build_transforms(cfg,is_train=True):
    if is_train:
        return A.Compose([
            A.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45,
                               p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomResizedCrop(cfg.INPUT.SIZE[0],cfg.INPUT.SIZE[1],p=1.0),
            A.Flip(p=0.5),
            A.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD,max_pixel_value=255.0,p=1.0),
            ToTensorV2(p=1.0)
        ],keypoint_params=A.KeypointParams(format='xy'),additional_targets={
            'tir':'image',
        })
    else:
        return A.Compose([
            A.Resize(cfg.INPUT.SIZE[0],cfg.INPUT.SIZE[1],p=1.0),
            A.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD,max_pixel_value=255.0,p=1.0),
            ToTensorV2(p=1.0)
        ],keypoint_params=A.KeypointParams(format='xy'),additional_targets={
            'tir':'image',
        })

def build(image_set,cfg):
    is_train=image_set=='train'
    root_rgb=cfg.RGB_DIR
    root_tir=cfg.TIR_DIR
    annFile=cfg.ANNOTATION_FILE
    transform_cfg=cfg.TRANSFORM
    transform=build_transforms(transform_cfg,is_train)
    return RGBTDataSet(root_rgb,root_tir,annFile,transform,local_rank=get_local_rank(),local_size=get_local_size())
        
    