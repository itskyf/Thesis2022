import time

import torch
import torch.nn.utils
import tqdm

from pcdet.utils import common_utils, commu_utils


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    model_fw_fn,
    lr_scheduler,
    accumulated_iter,
    optim_cfg,
    tbar,
    total_it_each_epoch,
    dataloader_iter,
    tb_writer=None,
    leave_pbar=False,
):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    pbar = (
        tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc="train", dynamic_ncols=True)
        if tb_writer is not None
        else None
    )
    data_time = common_utils.AverageMeter()
    batch_time = common_utils.AverageMeter()
    forward_time = common_utils.AverageMeter()

    model.train()
    for _ in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]["lr"]

        if tb_writer is not None:
            tb_writer.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)

        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_fw_fn(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if pbar is not None and tb_writer is not None:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update(
                {
                    "loss": loss.item(),
                    "lr": cur_lr,
                    "d_time": f"{data_time.val:.2f}({data_time.avg:.2f})",
                    "f_time": f"{forward_time.val:.2f}({forward_time.avg:.2f})",
                    "b_time": f"{batch_time.val:.2f}({batch_time.avg:.2f})",
                }
            )

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            tb_writer.add_scalar("train/loss", loss, accumulated_iter)
            tb_writer.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)
            for key, val in tb_dict.items():
                tb_writer.add_scalar("train/" + key, val, accumulated_iter)
    if pbar is not None:
        pbar.close()
    return accumulated_iter


def train_model(
    model,
    optimizer,
    train_loader,
    model_func,
    lr_scheduler,
    optim_cfg,
    start_epoch,
    total_epochs,
    start_iter,
    tb_log,
    ckpt_dir,
    lr_warmup_scheduler,
    save_interval: int,
):
    accumulated_iter = start_iter
    with tqdm.trange(
        start_epoch, total_epochs, desc="Epochs", dynamic_ncols=True, leave=tb_log is not None
    ) as tbar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            train_loader.sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model,
                optimizer,
                train_loader,
                model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                tbar=tbar,
                tb_writer=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if tb_log is not None and trained_epoch % save_interval == 0:
                ckpt_path = ckpt_dir / f"checkpoint_epoch_{trained_epoch}"
                torch.save(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                    f"{ckpt_path}.pth",
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }
