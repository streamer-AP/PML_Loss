import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import ColorJitter, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from misc.utils import get_local_rank, get_local_size
from torch.utils.data import Dataset

from .nwpu_crowd import NWPUCounting
from .torchvision_datasets.coco import CocoDetection


class CoCoCounting(CocoDetection):
    def __init__(self, root, annFile, mosaic=0, transforms=None, max_len=5000, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(root, annFile, transform=None, target_transform=None,
                         transforms=None, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.alb_transforms = transforms
        self.to_tensor = ToTensorV2()
        self.max_len = max_len
        self.mosaic = mosaic

    def __getitem__(self, index, load_image_mosaic=False):

        image, target = super().__getitem__(index)
        labels = {}

        w, h = image.size

        image = np.array(image)

        kps_with_classes = [(obj["keypoint"], obj["category_id"])
                            for obj in target if "keypoint" in obj]
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


def point_transform(image_set, image_scale=1024):
    if image_set == "train":
        return A.Compose([
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
            ColorJitter(),
            ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=15,
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
            # A.LongestMaxSize(2048),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=64,
                          pad_width_divisor=64, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True))


class MultiCounting(Dataset):
    def __init__(self, args, image_set):
        datasets = []
        for ann_file, img_prefix in zip(args.ann_files, args.img_prefixes):
            datasets.append(NWPUCounting(root=img_prefix,
                                         annFile=ann_file,
                                         max_len=args.max_len,
                                         transforms=point_transform(image_set),
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size()))
        self.to_tensor = ToTensorV2()
        self.datasets = datasets
        self.datasets_index = []
        self.datasets_offset = []
        dataset_idx = 0
        for dataset in self.datasets:
            for offset in range(len(dataset)):
                self.datasets_index.append(dataset_idx)
                self.datasets_offset.append(offset)
            dataset_idx += 1

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_idx = self.datasets_index[index]
        dataset_offset = self.datasets_offset[index]
        return self.datasets[dataset_idx][dataset_offset]


def build(image_set, args):
    dataset = MultiCounting(args, image_set)
    return dataset


def build_coco_counting(image_set, args):
    dataset = CoCoCounting(root=args.img_prefix,
                           annFile=args.ann_file,
                           max_len=args.max_len,
                           transforms=point_transform(
                               image_set, args.image_scale),
                           cache_mode=args.cache_mode,
                           local_rank=get_local_rank(),
                           local_size=get_local_size())
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
