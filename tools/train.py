import argparse
import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import List

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
import pcdet.models.detectors
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import model_fn_decorator


@record
def main():
    args, conf = parse_config()
    distributed.init_process_group(backend="nccl")
    torch.backends.cudnn.benchmark = True
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
    logger = create_logger(log_path, local_rank)

    logger.info("Total batch size: %d", distributed.get_world_size() * batch_size)
    log_config_to_file(conf, logger)
    if tb_writer is not None:
        shutil.copy(args.cfg_path, args.output_dir / args.cfg_path.name)

    train_set, train_loader = build_dataloader(
        conf.DATA_CONFIG, batch_size, args.num_workers, conf.CLASS_NAMES
    )

    model_fn = getattr(pcdet.models.detectors, conf.MODEL.NAME)
    model = model_fn(conf.MODEL, len(conf.CLASS_NAMES), train_set)
    optimizer = build_optimizer(model, conf.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = cur_it = 0
    last_epoch = -1
    if args.pretrained is not None:
        model.load_params_from_file(args.pretrained, logger)
    if args.ckpt is not None:
        cur_it, start_epoch = model.load_params_with_optimizer(args.ckpt, optimizer, logger)
    last_epoch = start_epoch + 1

    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader),
        total_epochs=epochs,
        last_epoch=last_epoch,
        optim_cfg=conf.OPTIMIZATION,
    )

    logger.info("Start training")
    with torch.cuda.device(local_rank):
        # -----------------------start training---------------------------
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
            tb_log=tb_writer,
            ckpt_dir=ckpt_dir,
            lr_warmup_scheduler=lr_warmup_scheduler,
            save_interval=args.save_interval,
        )


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--save_interval", type=int, default=5, help="number of training epochs")
    parser.add_argument(
        "--set", dest="set_cfgs", nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
    parser.add_argument("--start_epoch", type=int, default=0)
    # TODO eval in training
    parser.add_argument(
        "--num_epochs_to_eval", type=int, default=0, help="number of checkpoints to be evaluated"
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_path, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def create_logger(log_file, rank):
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


def build_dataloader(data_cfg, batch_size: int, num_workers: int, class_names: List[str]):
    train_set_fn = getattr(pcdet.datasets, data_cfg.DATASET)
    train_set = train_set_fn(data_cfg, class_names, training=True)
    train_sampler = ElasticDistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size,
        collate_fn=train_set.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    return train_set, train_loader


if __name__ == "__main__":
    main()
