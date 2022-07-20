from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from pcdet.datasets import load_data_to_gpu


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    scheduler,
    max_norm: float,
    tb_writer: SummaryWriter,
    epoch: int,
    ckpt_dir: Path,
):

    model.train()
    best_loss = float("inf")
    step = epoch * len(train_loader)
    for pcd_batch in tqdm(train_loader, leave=False, desc="Step", dynamic_ncols=True):
        pcd_batch = load_data_to_gpu(pcd_batch)

        loss_dict = model(pcd_batch)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        # log to console and tensorboard
        tb_writer.add_scalar("train/loss", loss, step)
        for key, val in loss_dict.items():
            tb_writer.add_scalar("train/" + key, val.item(), step)
        step += 1
        if (loss := loss.item()) < best_loss:
            best_loss = loss
            torch.save(
                _ckpt_state(model, optimizer, scheduler, epoch), ckpt_dir / "ckpt_best_loss.pt"
            )


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    scheduler,
    start_epoch: int,
    total_epochs: int,
    max_norm: float,
    tb_writer: SummaryWriter,
    ckpt_dir,
    save_interval: int,
):
    for epoch in trange(start_epoch, total_epochs, desc="Epochs", dynamic_ncols=True):
        # train one epoch
        # save trained model
        train_epoch(model, optimizer, train_loader, scheduler, max_norm, tb_writer, epoch, ckpt_dir)
        if tb_writer is not None and epoch % save_interval == save_interval - 1:
            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save(_ckpt_state(model, optimizer, scheduler, epoch), ckpt_path)


def _ckpt_state(model: nn.Module, optimizer: optim.Optimizer, scheduler, epoch: int):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
