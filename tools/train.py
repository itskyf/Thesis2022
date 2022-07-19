import argparse
import logging
import shutil
from pathlib import Path
from typing import List

import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_utils.train_utils import train_model

import pcdet.datasets
import pcdet.models
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file


def main():
    args, conf = parse_config()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")  # TODO

    batch_size = (
        conf.OPTIMIZATION.BATCH_SIZE_PER_GPU if args.batch_size is None else args.batch_size
    )
    total_epochs = conf.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(args.output_dir / "tensorboard")
    if tb_writer is not None:
        shutil.copy(args.cfg_path, args.output_dir / args.cfg_path.name)

    train_loader = build_dataloader(
        conf.DATA_CONFIG, batch_size, args.num_workers, conf.CLASS_NAMES
    )

    model_fn = getattr(pcdet.models, conf.MODEL.NAME)
    model: nn.Module = model_fn(conf.MODEL, len(conf.CLASS_NAMES))
    model.to(device)

    lr = conf.OPTIMIZATION.LR
    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, lr, epochs=total_epochs, steps_per_epoch=len(train_loader)
    )

    start_epoch = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    print("Total batch size:", batch_size)

    print("Start training, start epoch:", start_epoch)
    # -----------------------start training---------------------------
    train_model(
        model,
        optimizer,
        train_loader,
        scheduler,
        start_epoch,
        total_epochs,
        cfg.OPTIMIZATION.max_norm,
        tb_writer,
        ckpt_dir,
        args.save_interval,
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
    parser.add_argument("--save_interval", type=int, default=2)
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
    train_loader = DataLoader(
        train_set,
        batch_size,
        collate_fn=train_set.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader


if __name__ == "__main__":
    main()
