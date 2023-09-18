from .counting_loss import DenseMap_Loss

def build_loss(cfg):
    if cfg.type=="dmap_loss":
        return DenseMap_Loss(cfg.args)