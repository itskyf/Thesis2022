from functools import partial

import torch.optim.lr_scheduler as lr_sched
from torch import nn, optim

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


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

        def children(module: nn.Module):
            return list(module.children())

        def num_children(module: nn.Module) -> int:
            return len(children(module))

        flatten_model = (
            lambda module: sum(map(flatten_model, module.children()), [])
            if num_children(module)
            else [module]
        )

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


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
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
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer,
                T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR,
            )

    return lr_scheduler, lr_warmup_scheduler
