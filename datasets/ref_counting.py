
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np

import cv2

from .crowd_counting_datasets import CoCoCounting
from .torchvision_datasets.coco import CocoDetection


def fsc_transform(image_set, image_scale=1024):
    if image_set == "train":
        return A.Compose([
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
            A.ColorJitter(),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=15,
                               p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomResizedCrop(image_scale, image_scale, scale=(1, 1), p=1),
            A.HorizontalFlip(),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True)
        )
    elif image_set == "val" or image_set == "test":
        return A.Compose([
            A.Resize(image_scale, image_scale),
            A.Normalize(),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))

def patch_transform(image_scale=128):
    return A.Compose([
        A.Resize(image_scale, image_scale),
        A.Normalize(),
    ])

class CoCoCounting(CocoDetection):
    def __init__(self, root, annFile, mosaic=0, transforms=None,patch_scale=128,max_patch=10, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.patch_transforms = patch_transform(patch_scale)
        self.to_tensor = ToTensorV2()
        self.max_len = max_len
        self.max_patch = max_patch
        self.patch_scale=patch_scale
        self.mosaic = mosaic

    def __getitem__(self, index, load_image_mosaic=False):

        image, target = super().__getitem__(index)
        labels = {}

        w, h = image.size
        image = np.array(image)
        kps_with_classes = [(obj["keypoint"], obj["category_id"])
                            for obj in target if "keypoint" in obj]
        bbox_with_classes = [obj["bbox"] for obj in target if "bbox" in obj][:self.max_patch]
        patches_labels=torch.zeros((self.max_patch,3,self.patch_scale,self.patch_scale))
        for idx,bbox in enumerate(bbox_with_classes):
            patch=image[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
            

            patch=self.patch_transforms(image=patch)["image"]

            patch_tensor=self.to_tensor(image=patch)["image"]
            patches_labels[idx]=patch_tensor
        labels["patch"]=patches_labels
        labels["patch_num"]=len(bbox_with_classes)
        labels["raw_num"] = len(kps_with_classes)
        clses, kpses = [], []
        # filter out the out of range image
        for pt, cls in kps_with_classes:
            if pt[0] > w-1 or pt[1] > h-1 or pt[0] < 0 or pt[1] < 0:
                continue
            else:
                clses.append(cls)
                kpses.append(pt)
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

def build(image_set,args):
    return CoCoCounting(args.img_prefix, args.ann_file, mosaic=False,
                        transforms=fsc_transform(image_set,args.image_scale),
                        patch_scale=args.patch_scale,
                        max_patch=args.max_patch,
                        max_len=args.max_len,
                        cache_mode=args.cache_mode,)
                        