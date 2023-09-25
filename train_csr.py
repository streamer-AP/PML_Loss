import argparse
import json
import logging
import os

from pprint import pprint

import torch
from easydict import EasyDict as edict
from termcolor import cprint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from eingine.trainer import evaluate_counting, train_one_epoch
from misc import utils
from misc.drawer import Drawer_DenseMap
from misc.saver_builder import Saver
from misc.utils import MetricLogger,is_main_process
# from models.sanet import SANet
from models.CSRNet import CSRNet

from models.loss import build_loss
from optimizer import loss_weight_builder, optimizer_builder, scheduler_builder

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        k = k[11:]
        state_dict[k] = v
    return state_dict

def main(args):
    utils.init_distributed_mode(args)
    utils.set_randomseed(42 + utils.get_rank())

    # initilize the model
    model = model_without_ddp = CSRNet()

    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build the dataset and dataloader
    dataset_train = build_dataset(image_set='train', args=args.Dataset.train)
    dataset_val = build_dataset(image_set='val', args=args.Dataset.val)
    sampler_train = DistributedSampler(
        dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(
        dataset_val, shuffle=False) if args.distributed else None
    loader_train = DataLoader(dataset_train, batch_size=args.Dataset.train.batch_size, sampler=sampler_train, shuffle=(
        sampler_train is None), num_workers=0, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=args.Dataset.val.batch_size, sampler=sampler_val,
                            shuffle=False, num_workers=0, pin_memory=True)
    optimizer = optimizer_builder(args.Optimizer, model_without_ddp)
    scheduler = scheduler_builder(args.Scheduler, optimizer)
    if args.Scheduler.ema:
        def ema_avg(averaged_model_parameter, model_parameter, num_averaged): return \
            args.Scheduler.ema_weight * averaged_model_parameter + \
            (1-args.Scheduler.ema_weight) * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        ema_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                                    anneal_strategy=args.Scheduler.ema_annel_strategy, anneal_epochs=args.Scheduler.ema_annel_epochs, swa_lr=args.Scheduler.ema_lr)
        ema_saver = Saver(args.Saver, is_ema=True)
        ema_drawer = Drawer_DenseMap(args.Drawer, is_ema=True)
    else:
        ema_model = None
        ema_scheduler = None
    loss_weight = loss_weight_builder(args.Loss_Weight)
    counting_criterion = build_loss(args.Loss.counting)
    saver = Saver(args.Saver)
    drawer = Drawer_DenseMap(args.Drawer)
    if args.Misc.use_tensorboard:
        tensorboard_writer = SummaryWriter(args.Misc.tensorboard_dir)
    for epoch in range(args.Misc.epochs):
        logging.info('epoch: {}'.format(epoch))
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_logger = MetricLogger(args.Logger)
        val_logger=MetricLogger(args.Logger)
        stats = edict()
        drawer.clear()
        stats.train_stats = train_one_epoch(
            model, counting_criterion,loader_train, optimizer, train_logger, drawer, loss_weight, epoch, args)
        if args.Scheduler.ema and epoch > args.Scheduler.ema_start_epoch:
            ema_model.update_parameters(model)
            ema_scheduler.step()
            torch.optim.swa_utils.update_bn(loader_train, ema_model)
            stats.ema_test_stats = evaluate_counting(
                ema_model, counting_criterion, loader_val, val_logger, ema_drawer, epoch, args)
            ema_saver.save_on_master(
                ema_model, optimizer, scheduler, epoch, stats)
        else:
            scheduler.step()
            stats.ema_test_stats = {}
            stats.ema_test_stats = {}
        drawer.clear()
        stats.test_stats = evaluate_counting(
            model, counting_criterion, loader_val, val_logger, drawer, epoch, args)
        saver.save_on_master(model, optimizer, scheduler, epoch, stats)
        log_stats = {**{f'train_{k}': v for k, v in stats.train_stats.items()},
                     **{f'val_{k}': v for k, v in stats.test_stats.items()},
                     **{f'ema_val_{k}': v for k, v in stats.ema_test_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            for key, value in log_stats.items():
                cprint(f'{key}:{value}', 'green')
                logging.info(f'{key}:{value}')
                if args.Misc.use_tensorboard:
                    tensorboard_writer.add_scalar(key, value, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Counting Model Trainer ")
    parser.add_argument(
        "--config", default="")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    print(is_main_process())
    if is_main_process():
        os.makedirs(cfg.Saver.save_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(cfg.Saver.save_dir, "log.txt"),
                            level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s', filemode="a")
        json.dump(cfg, open(os.path.join(
            cfg.Saver.save_dir, "config.json"), "w"), indent=4)
        pprint(cfg)
    main(cfg)
