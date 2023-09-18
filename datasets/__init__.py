# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .crowd_human import build as build_crowd_human
from .nwpu_crowd import build as build_nwpu_crowd
from .nwpu_crowd import show_example as show_nwpu_crowd_dataset
from .crowd_counting_datasets import build as build_multi_crowd
from .crowd_counting_datasets import build_coco_counting
from .ref_counting import build as build_ref_counting
from .dronergbt import build as build_rgbt_counting
from .st_crowd import build as build_st_counting
from .uavcc import build as build_uavcc
from .image_folder import ImageFolder
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.name == 'coco':
        return build_coco(image_set, args)
    if args.name == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.name=="ref_counting":
        return build_ref_counting(image_set,args)
    if args.name=="rgbt_counting":
        return build_rgbt_counting(image_set,args)
    if args.name=="crowd_human":
        return build_crowd_human(image_set,args)
    if args.name=="coco_counting":
        return build_coco_counting(image_set,args)
    if args.name=="nwpu_crowd":
        return build_nwpu_crowd(image_set,args)
    if args.name=="st_crowd":
        return build_st_counting(image_set,args)
    if args.name=="multi_crowd":
        return build_multi_crowd(image_set,args)
    if args.name=="uavcc":
        return build_uavcc(image_set,args)
    raise ValueError(f'dataset {args.name} not supported')

def show_example(dataset,args,index):
    if args.name=="nwpu_crowd":
        show_nwpu_crowd_dataset(dataset,index)