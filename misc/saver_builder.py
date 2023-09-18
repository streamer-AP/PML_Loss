import logging
import torch
import os
from .utils import is_main_process


class Saver():
    def __init__(self, args, is_ema=False) -> None:
        self.save_dir = args.save_dir
        self.save_interval = args.save_interval
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_best = args.save_best
        self.metric=args.metric
        self.best_metric = 1e10
        self.is_ema = "ema" if is_ema else "normal"

    def save(self, model, optimizer, scheduler, filename, epoch, stats={}):
        if is_main_process():
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'states': stats
            }, os.path.join(self.save_dir, filename))

    def save_inter(self, model, optimizer, scheduler, name, epoch, stats={}):
        if epoch % self.save_interval == 0:
            self.save(model, optimizer, scheduler, f"{name}.pth", epoch, stats)

    def save_on_master(self, model, optimizer, scheduler, epoch, stats={}):
        if is_main_process():
            self.save_inter(model, optimizer, scheduler,
                            f"checkpoint{epoch:04}_{self.is_ema}.pth", epoch, stats)
            logging.info(f"save checkpoint{epoch:04}_{self.is_ema}")
            if self.is_ema == "ema":
                if self.save_best and stats.ema_test_stats[self.metric] < self.best_metric:
                    self.best_metric = stats.ema_test_stats[self.metric]
                    self.save(model, optimizer, scheduler,
                              f"best_{self.is_ema}.pth", epoch, stats)
                    logging.info(
                        f"save best ema model,mae:{self.best_metric},epoch:{epoch}")

            else:
                if self.save_best and stats.test_stats[self.metric] < self.best_metric:
                    self.best_metric = stats.test_stats[self.metric]
                    logging.info(
                        f"save best model,{self.metric}:{self.best_metric},epoch:{epoch}")
                    self.save(model, optimizer, scheduler,
                              f"best_{self.is_ema}.pth", epoch, stats)

    def save_last(self, model, optimizer, scheduler, epoch, stats={}):
        if is_main_process():
            self.save(model, optimizer, scheduler,
                      f"checkpoint_last.pth", epoch, stats)
