import argparse
import datetime
import functools
import logging
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.backends.cudnn
from torch import distributed, nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

import pcdet.datasets
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network, model_fn_decorator


@record
def main():
    args, conf = _parse_config()
    distributed.init_process_group(backend="nccl")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    local_rank = int(os.environ["LOCAL_RANK"])

    batch_size = (
        conf.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    )
    epochs = conf.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(args.output_dir / "tensorboard") if local_rank == 0 else None
    log_path = (
        args.output_dir / f"log_train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    logger = _create_logger(log_path, local_rank)

    logger.info("Total_batch_size: %d", distributed.get_world_size() * batch_size)
    for key, val in vars(args).items():
        logger.info("%-16s %s", key, val)
    log_config_to_file(conf, logger=logger)
    if tb_writer is not None:
        (args.output_dir / args.cfg_path.name).write_text(args.cfg_path.read_text())

    train_set, train_loader = _build_dataloader(
        conf.DATA_CONFIG, batch_size, args.num_workers, args.seed, conf.CLASS_NAMES
    )

    model = build_network(model_cfg=conf.MODEL, num_class=len(conf.CLASS_NAMES), dataset=train_set)
    optimizer = build_optimizer(model, conf.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = cur_it = 0
    last_epoch = -1
    if args.pretrained is not None:
        model.load_params_from_file(args.pretrained, logger)
    if args.ckpt is not None:
        cur_it, start_epoch = model.load_params_with_optimizer(args.ckpt, optimizer, logger)
    last_epoch = start_epoch + 1

    # TODO is that?
    with torch.cuda.device(local_rank):
        model.cuda()
        model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        logger.info(model)

        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer,
            total_iters_each_epoch=len(train_loader),
            total_epochs=epochs,
            last_epoch=last_epoch,
            optim_cfg=conf.OPTIMIZATION,
        )

        # -----------------------start training---------------------------
        logger.info("Start training")
        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=conf.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=epochs,
            start_iter=cur_it,
            rank=local_rank,
            tb_log=tb_writer,
            ckpt_save_dir=ckpt_dir,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
        )


def _parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--seed", default=2912, type=int)
    parser.add_argument(
        "--ckpt_save_interval", type=int, default=5, help="number of training epochs"
    )
    parser.add_argument(
        "--max_ckpt_save_num", type=int, default=30, help="max number of saved checkpoint"
    )
    parser.add_argument(
        "--set", dest="set_cfgs", nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
    parser.add_argument("--max_waiting_mins", type=int, default=0, help="max waiting minutes")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument(
        "--num_epochs_to_eval", type=int, default=0, help="number of checkpoints to be evaluated"
    )
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_path, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def _create_logger(log_file, rank):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _build_dataloader(
    data_cfg, batch_size: int, num_workers: int, seed: int, class_names: List[str]
):
    train_set_fn = getattr(pcdet.datasets, data_cfg.DATASET)
    train_set = train_set_fn(data_cfg, class_names, training=True)
    train_sampler = ElasticDistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_set.collate_batch,
        sampler=train_sampler,
        timeout=0,
        worker_init_fn=functools.partial(_worker_init_fn, seed=seed),
    )
    return train_set, train_loader


def _worker_init_fn(worker_id: int, seed: int):
    seed += worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
