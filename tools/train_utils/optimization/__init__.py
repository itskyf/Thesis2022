from functools import partial

from torch import nn, optim

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def flatten_model(module: nn.Module):
    return sum(map(flatten_model, module.children()), []) if list(module.children()) else [module]


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == "adam":
        return optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    if optim_cfg.OPTIMIZER == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=optim_cfg.LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM,
        )
    if optim_cfg.OPTIMIZER == "adam_onecycle":
        optimizer_func = partial(optim.AdamW, betas=(0.9, 0.99))
        return OptimWrapper.create(
            optimizer_func,
            3e-3,
            [nn.Sequential(*flatten_model(model))],  # get layer groups
            wd=optim_cfg.WEIGHT_DECAY,
            true_wd=True,
            bn_wd=True,
        )
    raise NotImplementedError


def build_scheduler(optimizer, total_epochs, total_iters_each_epoch, optim_cfg):
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == "adam_onecycle":
        lr_scheduler = OneCycle(
            optimizer,
            total_steps,
            optim_cfg.LR,
            list(optim_cfg.MOMS),
            optim_cfg.DIV_FACTOR,
            optim_cfg.PCT_START,
        )
    else:
        raise NotImplementedError
    return lr_scheduler
