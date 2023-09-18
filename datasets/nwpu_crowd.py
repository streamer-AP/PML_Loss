import os
from random import random
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import (ColorJitter, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from misc.utils import get_local_rank, get_local_size
from .torchvision_datasets.coco import CocoDetection


def bbox_transform(image_set):
    if image_set == "train":
        return A.Compose([
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
            ColorJitter(),
            ShiftScaleRotate(shift_limit=0, scale_limit=0.5, rotate_limit=15,
                             p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomResizedCrop(1024, 1024, scale=(1, 1), p=1),
            A.CoarseDropout(max_holes=100, max_height=128, max_width=128, min_height=64, min_width=64, p=0.5),
            A.HorizontalFlip(),
            A.Normalize(),
        ],
            bbox_params=A.BboxParams(format='coco', label_fields=[
                'class_labels'])
        )
    elif image_set == "val":
        return A.Compose([
            A.LongestMaxSize(1536),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=64,
                          pad_width_divisor=64, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.2))


def point_transform(image_set,size):
    if image_set == "train":
        return A.Compose([
            A.GaussNoise(p=0.2),
            ColorJitter(),
            A.ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45,
                             p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(),

            A.RandomResizedCrop(size, size, scale=(1, 1), p=1),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True)
        )
    elif image_set == "val":
        return A.Compose([
            A.LongestMaxSize(size),
            A.PadIfNeeded(min_height=size, min_width=size,  border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))
    else:
        return A.Compose([
            # A.LongestMaxSize(size),

            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=size,
                          pad_width_divisor=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))
class NWPUDectecting(CocoDetection):
    def __init__(self, root, annFile, transforms=None, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len
    def __getitem__(self, index):
        image,target=super().__getitem__(index)
        w,h=image.size
        
        image=np.array(image)
        bboxes_with_classes = [(obj["bbox"], obj["category_id"])
                               for obj in target]
        bboxes, clses = [], []
        
        for bbox,cls in bboxes_with_classes:
            if bbox[0]+bbox[2]/2 > w or bbox[1]+bbox[3]/2 > h:
                continue
            else:
                clses.append(cls)
                bboxes.append((max(0,bbox[0]),max(0,bbox[1]),min(w-bbox[0],bbox[2]),min(h-bbox[1],bbox[3])))
        data=self.alb_transforms(image=image,bboxes=bboxes,class_labels=clses)
        labels={}
        labels["num"] = torch.as_tensor(
            len(data["bboxes"]), dtype=torch.long)

        bboxes= torch.as_tensor(
            data["bboxes"], dtype=torch.float32)
        clses = torch.as_tensor(
            data["class_labels"], dtype=torch.long)
        image=data["image"]
        image = self.to_tensor(image=image)["image"]
        labels["bboxes"] = torch.zeros((self.max_len, 4), dtype=torch.float32)
        labels["classes"] = torch.zeros((self.max_len), dtype=torch.long)
        if labels["num"] > 0:
            labels["bboxes"][:bboxes.shape[0]] = bboxes[:self.max_len]
            labels["classes"][:clses.shape[0]] = clses[:self.max_len]
        if labels["num"] > self.max_len:
            print(
                f"Warning: the number of points {labels['num']} is larger than max_len {self.max_len}")
            labels["num"] = torch.as_tensor(self.max_len, dtype=torch.long)
        return image, labels
        
        
    
class NWPUCounting(CocoDetection):
    def __init__(self, root, annFile, mosaic=0,transforms=None, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len
        self.mosaic=mosaic
            
    def __getitem__(self, index,load_image_mosaic=False):
        if self.mosaic>0 and random.random()<self.mosaic:
            image, target = super().__getitem__(index)
            
            images,targets=[image],[target]
            for i in range(3):
                image,target=super().__getitem__(random.randint(0,len(self.coco.imgs)-1))
                
        image, target = super().__getitem__(index)
        labels = {}
        
        w, h = image.size

        image = np.array(image)
        bboxes_with_classes = [(obj["bbox"], obj["category_id"])
                            for obj in target]
        labels["raw_num"]=len(bboxes_with_classes)
        clses, kpses = [], []
        # filter out the out of range bboxes
        for bbox, cls in bboxes_with_classes:
            if bbox[0]+bbox[2]/2 > w-1 or bbox[1]+bbox[3]/2 > h-1 or bbox[0]<0 or bbox[1]<0:
                continue
            else:
                clses.append(cls)
                kpses.append((bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2))
        data = self.alb_transforms(
            image=image, keypoints=kpses, class_labels=clses)
        image = data["image"]
        keep = [idx for idx, v in enumerate(data["keypoints"]) if v[1] < (
            image.shape[0]-1) and v[0] < (image.shape[1]-1)]
        labels["num"] = torch.as_tensor(
            len(keep), dtype=torch.long)

        kpses = torch.as_tensor(
            data["keypoints"], dtype=torch.float32)[keep]
        clses = torch.as_tensor(
            data["class_labels"], dtype=torch.long)[keep]
        assert kpses.shape[0] == clses.shape[0], f"{kpses.shape[0]},{clses.shape[0]}"
        image = self.to_tensor(image=image)["image"]
        labels["points"] = torch.zeros((self.max_len, 2), dtype=torch.float32)
        labels["labels"] = torch.zeros((self.max_len), dtype=torch.long)
        if labels["num"] > 0:
            labels["points"][:kpses.shape[0]] = kpses[:self.max_len]
            labels["labels"][:clses.shape[0]] = clses[:self.max_len]
        if labels["num"] > self.max_len:
            print(
                f"Warning: the number of points {labels['num']} is larger than max_len {self.max_len}")
            labels["num"] = torch.as_tensor(self.max_len, dtype=torch.long)
        return image, labels


def build(image_set, args):
    img_prefix = args.img_prefix
    ann_file = args.ann_file
    assert os.path.exists(ann_file), f"annotation file {ann_file} not exists"
    if args.task=="counting":
        dataset = NWPUCounting(img_prefix, ann_file, max_len=args.max_len, transforms=point_transform(image_set,args.size),
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif args.task=="detection":
        dataset=NWPUDectecting(img_prefix, ann_file, max_len=args.max_len, transforms=bbox_transform(image_set,args.size),
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    else:
        raise NotImplementedError
    return dataset


def show_example(dataset, index):
    image, labels = dataset[index]
    print(f"labels: {labels}")
    print(f"image: {image.shape}")
    import matplotlib.pyplot as plt
    image = image.permute(1, 2, 0).numpy()
    for pts in labels["points"]:
        x, y = pts
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
