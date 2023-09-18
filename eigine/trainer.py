import imp
import math
import os
from typing import Iterable

import cv2
import einops
import torch

from .utils import (SmoothedValue, get_total_grad_norm, is_main_process,
                    reduce_dict)

def train_one_epoch(model: torch.nn.Module, counting_criterion: torch.nn.Module, locater_criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, metric_logger: object, drawer: object, loss_weight: object, epoch, args):
    model.train()
    counting_criterion.train()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    for inputs, labels in metric_logger.log_every(data_loader):
        optimizer.zero_grad()

        inputs = inputs.to(args.gpu)
        output_aux_dmap = model(inputs)
        all_loss = 0
        loss_dict = {}
        
        for j, dmap in enumerate( output_aux_dmap):
            aux_scale = inputs.shape[-1]/dmap.shape[-1]
            aux_counting_loss, aux_dMaps = counting_criterion(
                dmap, labels, aux_scale)
            for key, value in aux_counting_loss.items():
                loss_dict[f"aux_counting_{j}_{key}"] = value
            all_loss += aux_counting_loss["all"]
        loss_dict["all"] = all_loss

        loss_dict_reduced = reduce_dict(loss_dict)
        all_loss_reduced = loss_dict_reduced["all"]
        loss_value = all_loss_reduced.item()

        all_loss.backward()
        if args.Misc.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.Misc.clip_max_norm)
        else:
            grad_total_norm = get_total_grad_norm(
                model.parameters(), args.Misc.clip_max_norm)
        optimizer.step()
        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.update(loss_weight=loss_weight(epoch))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_counting(model, counting_criterion, locating_criterion, data_loader, metric_logger, drawer, epoch, args):
    model.eval()
    counting_criterion.eval()
    locating_criterion.eval()
    header = "Test"
    metric_logger.set_header(header)
    model.only_dmap = True
    with torch.no_grad():
        for inputs, labels in metric_logger.log_every(data_loader):
            B = inputs.shape[0]
            assert B == 1, "Batch size must be 1"
            inputs = inputs.to(args.gpu)

            # output_aux_dmap = model(inputs)
            output_aux_dmap = model(inputs)[-1]
            
            target_nums = labels["num"].to(args.gpu).float()
            metrics={}
            scale = inputs.shape[-1]/output_aux_dmap.shape[-1]
            predict_nums = torch.sum(output_aux_dmap, (1, 2, 3))

            mae = torch.abs(predict_nums-target_nums).data
            rmse = torch.pow(predict_nums-target_nums, 2).data

            nae = torch.abs(predict_nums - target_nums).data / \
                (max(1, target_nums.data))
            metrics[f"mae"] = mae
            metrics[f"rmse"] = rmse
            metrics[f"nae"] = nae
            metric_logger.update(**metrics)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k,
                 meter in metric_logger.meters.items()}
        stats["rmse"] = math.sqrt(stats["rmse"])
    return stats
