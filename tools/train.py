import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.backends.cudnn
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from torch import distributed, nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm, trange

from pcdet.datasets import IDataset, PCBatch
from pcdet.utils import resolver

logger: logging.Logger = logging.getLogger(__name__)


@record
@hydra.main(version_base=None, config_path=None, config_name="main")
def main(cfg: DictConfig):
    distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.enabled = True

    log_dir = Path(cfg.log_dir)
    tb_writer = SummaryWriter(cfg.log_dir) if local_rank == 0 else None
    if tb_writer is not None:
        logger.info("Tensorboard directory: %s", tb_writer.log_dir)
        logger.info("Batch size per GPU: %d", cfg.batch_size)
        logger.info("World size: %d", distributed.get_world_size())
    else:
        logger.setLevel(logging.WARNING)

    train_loader = initialize_data_loader(cfg.dataset, cfg.batch_size, cfg.num_workers)

    train_loader_len = len(train_loader)
    total_steps = train_loader_len * cfg.epochs
    model, optimizer, lr_scheduler = initialize_model(
        cfg.model, cfg.optim, cfg.scheduler, device, total_steps
    )

    try:
        cp_path = Path(cfg.checkpoint)
    except ConfigAttributeError:
        cp_path = None
    state = load_checkpoint(model, optimizer, device, cp_path)

    log_interval = train_loader_len // 5
    start_epoch = state.epoch + 1
    logger.info("Start epoch: %d", start_epoch)

    for epoch in trange(
        start_epoch,
        cfg.epochs,
        desc="Epoch",
        disable=tb_writer is None,
        dynamic_ncols=True,
    ):
        state.epoch = epoch
        global_step = epoch * train_loader_len
        train_loader.sampler.set_epoch(epoch)

        with torch.cuda.device(device):
            epoch_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                max_norm=cfg.grad_max_norm,
                global_step=global_step,
                device=device,
                tb_writer=tb_writer,
                log_interval=log_interval,
            )

        if tb_writer is not None:
            if epoch % 3 == 1:
                state.save(log_dir / f"ckpt_epoch_{epoch}.pt")
            if epoch_loss < state.min_loss:
                state.min_loss = epoch_loss
                state.save(log_dir / f"ckpt_loss_{epoch_loss}.pt")


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model: nn.parallel.DistributedDataParallel, optimizer: optim.Optimizer):
        self.epoch = -1
        self.min_loss = float("inf")
        self.model = model
        self.optimizer = optimizer

    def _apply_snapshot(self, snapshot):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = snapshot["epoch"]
        self.min_loss = snapshot["min_loss"]
        self.model.load_state_dict(snapshot["state_dict"])
        self.optimizer.load_state_dict(snapshot["optimizer"])

    def save(self, path: Path):
        snapshot = {
            "epoch": self.epoch,
            "min_loss": self.min_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(snapshot, path)

    def load(self, path: Path, device: torch.device):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(path, map_location=device)
        self._apply_snapshot(snapshot)


def load_checkpoint(
    model: nn.parallel.DistributedDataParallel,
    optimizer: optim.Optimizer,
    device: torch.device,
    path: Optional[Path],
):
    state = State(model, optimizer)
    if path is not None:
        logger.info("Loading checkpoint %s", path)
        state.load(path, device)
        logger.info("Loaded checkpoint %s", path)
    return state


def initialize_data_loader(data_cfg: DictConfig, batch_size: int, num_workers: int):
    train_set: IDataset = hydra.utils.instantiate(data_cfg, split="train")
    train_sampler = ElasticDistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size,
        collate_fn=train_set.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    return train_loader


def initialize_model(
    model_cfg: DictConfig,
    optim_cfg: DictConfig,
    scheduler_cfg: DictConfig,
    device: torch.device,
    total_steps: int,
) -> Tuple[nn.parallel.DistributedDataParallel, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    model: nn.Module = hydra.utils.instantiate(model_cfg)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    optimizer: optim.Optimizer = hydra.utils.instantiate(optim_cfg, params=model.parameters())
    # TODO others lr_scheduler
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=total_steps, **scheduler_cfg
    )
    return model, optimizer, lr_scheduler


def train_epoch(
    model: nn.parallel.DistributedDataParallel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    max_norm: float,
    global_step: int,
    device: torch.device,
    tb_writer: Optional[SummaryWriter],
    log_interval: int,
) -> float:
    model.train()
    sum_loss = 0
    for pc_batch in tqdm(train_loader, desc="Step", disable=tb_writer is None, dynamic_ncols=True):
        global_step += 1
        optimizer.zero_grad()
        pc_batch: PCBatch = pc_batch.to(device)

        loss_dict: Dict[str, torch.Tensor] = model(pc_batch)
        loss: torch.Tensor = sum([val for val in loss_dict.values() if val is not None])
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()

        sum_loss += loss.item()
        if tb_writer is not None and global_step % log_interval == log_interval - 1:
            tb_writer.add_scalar("train/loss", loss, global_step)
            for key, loss in loss_dict.items():
                tb_writer.add_scalar(f"train/{key}", loss, global_step)
    return sum_loss / len(train_loader)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_grid_size", resolver.get_grid_size)
    OmegaConf.register_new_resolver("len", lambda arr: len(arr))
    main()
