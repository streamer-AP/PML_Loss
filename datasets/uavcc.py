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
class UAVCCDataSet(Dataset):
    def __init__(self,root,annFile,transforms,max_len=5000,local_rank=0,local_size=1):
        super().__init__()
        self.transforms=transforms
        self.max_len=max_len
        self.local_rank=local_rank
        self.local_size=local_size
        self.root=root
        with open(annFile,'r') as f:
            self.anns=json.load(f)
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self,idx):
        ann=self.anns[idx]
        rgb_path=os.path.join(self.root,ann['filename'])
        rgb=cv2.imread(rgb_path)

        h,w=rgb.shape[:2]
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        bboxes=ann["bboxes"]
        keypoints=[[int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2)] for bbox in bboxes]
        keypoints_in_image=[]
        for pt in keypoints:
            if pt[0]>=w or pt[1]>=h or pt[0]<0 or pt[1]<0:
                continue
            else:
                keypoints_in_image.append(pt)
        keypoints=keypoints_in_image
        if self.transforms is not None:
            data=self.transforms(image=rgb,keypoints=keypoints)
            rgb=data['image']
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
        labels["filename"]=ann["filename"]
        labels["filepath"]=rgb_path
        return rgb,labels


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
        ],keypoint_params=A.KeypointParams(format='xy'))
    else:
        return A.Compose([
            A.Resize(cfg.INPUT.SIZE[0],cfg.INPUT.SIZE[1],p=1.0),
            A.Normalize(mean=cfg.INPUT.PIXEL_MEAN,std=cfg.INPUT.PIXEL_STD,max_pixel_value=255.0,p=1.0),
            ToTensorV2(p=1.0)
        ],keypoint_params=A.KeypointParams(format='xy'))

def build(image_set,cfg):
    is_train=cfg.type=='train'
    root=cfg.img_prefix
    annFile=cfg.ann_file
    transform_cfg=cfg.TRANSFORM
    transform=build_transforms(transform_cfg,is_train)
    return UAVCCDataSet(root,annFile,transform,local_rank=get_local_rank(),local_size=get_local_size())
        
    