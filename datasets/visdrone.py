import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from misc.utils import get_local_rank, get_local_size

import datasets.transforms as T

from .torchvision_datasets.coco import CocoDetection
import einops
## to be implemented
def make_transform(image_set):
    if image_set == "train":
        return A.Compose([
            A.ColorJitter(),
            A.Rotate(45,border_mode=cv2.BORDER_CONSTANT,value=0),
            A.RandomResizedCrop(512, 512, scale=(0.33, 1.5), p=1),
            A.HorizontalFlip(),
            A.GaussNoise(p=0.1),
            A.Normalize(),
        ],
            keypoint_params=A.KeypointParams(format='xy', label_fields=[
                                             'class_labels'], remove_invisible=True),
            #bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.1)
        )
    elif image_set == "val":
        return A.Compose([
            A.LongestMaxSize(1024),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=64,
                          pad_width_divisor=64, border_mode=cv2.BORDER_DEFAULT, value=0),
            A.Normalize(),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))


class CocoCounting(CocoDetection):
    def __init__(self, root, annFile, transforms=None, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor=ToTensorV2()
    def __getitem__(self, index):

        image, target = super().__getitem__(index)
        w, h = image.size

        image = np.array(image)
        anno = target
        image_id = self.ids[index]
        image_id = torch.tensor([image_id])
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        classes = [obj["category_id"] for obj in anno]
        boxes = [(box, c) for c, box in zip(classes, boxes)
                 if box[0]+box[2] < w and box[0]>0 and box[1]+box[3] < h and box[1]>0]
        classes = [v[1] for v in boxes]
        boxes = [v[0] for v in boxes]
        kps = [(box[0]+box[2]/2, box[1]+box[3]/2) for box in boxes]
        data = self.alb_transforms(
            image=image, keypoints=kps, class_labels=classes)
        image = data["image"]
        target = {}
        keep=[idx for idx,v in enumerate(data["keypoints"]) if v[1]<(image.shape[0]-1) and v[0]<(image.shape[1]-1)]
        target["num"] = torch.as_tensor(
            len(keep), dtype=torch.float32)

        target["points"] = torch.as_tensor(
            data["keypoints"], dtype=torch.float32)[keep]
        target["class"] = torch.as_tensor(
            data["class_labels"], dtype=torch.long)[keep]
        assert target["points"].shape[0]==target["class"].shape[0],f"{target['points'].shape[0]},{target['class'].shape[0]}"
        image=self.to_tensor(image=image)["image"]
        return image, target


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    if os.path.exists(os.path.join(root, args.train_ann_path)):
        train_path = os.path.join(root, args.train_ann_path)
    elif os.path.exists(os.path.join(root, 'annotation_train.json')):
        train_path = os.path.exist(os.path.join(root, 'annotation_train.json'))
    else:
        raise ValueError(
            f"cannot find train annotation file path under {root} ")
    if os.path.exists(os.path.join(root, args.val_ann_path)):
        val_path = os.path.join(root, args.val_ann_path)
    elif os.path.exists(os.path.join(root, 'annotation_val.json')):
        val_path = os.path.exist(os.path.join(root, 'annotation_val.json'))
    else:
        raise ValueError(f"cannot find val annotation file path under {root} ")
    PATHS = {
        "train": (root / "images", train_path),
        "val": (root / "images", val_path),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoCounting(img_folder, ann_file, transforms=make_transform(image_set),
                           cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
