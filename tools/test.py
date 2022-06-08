import itertools
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import hydra
import torch
import torch.backends.cudnn
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from torch import distributed, nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcdet.datasets import FlatDistSampler, IDataset, PCBatch
from pcdet.models import ModelOutput
from pcdet.utils import resolver

logger: logging.Logger = logging.getLogger(__name__)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@record
@hydra.main(version_base=None, config_path=None, config_name="main")
def main(cfg: DictConfig):
    distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.enabled = True

    if local_rank == 0:
        logger.info("Batch size per GPU: %d", cfg.batch_size)
        logger.info("World size: %d", distributed.get_world_size())
    else:
        logger.setLevel(logging.WARNING)

    val_loader, val_set = initialize_data(cfg.dataset, cfg.batch_size, cfg.num_workers)
    try:
        det_path = Path(cfg.det_path)
        with det_path.open("rb") as det_file:
            det_annos = pickle.load(det_file)
        get_evaluation_result(det_annos, val_set)
        # TODO better handling
    except ConfigAttributeError:
        model: nn.Module = hydra.utils.instantiate(cfg.model)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

        recall_thresholds = cfg.model.recall_thresholds
        ckpt_path = Path(cfg.checkpoint)
        with torch.cuda.device(device):
            if ckpt_path.is_dir():
                raise NotImplementedError
            else:
                load_checkpoint(model, ckpt_path, device)
                evaluate_checkpoint(
                    model, val_loader, val_set, device, recall_thresholds, ckpt_path
                )


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.parallel.DistributedDataParallel,
    val_loader: DataLoader,
    val_set: IDataset,
    device: torch.device,
    recall_thresholds: List[float],
    ckpt_path: Path,
):
    model.eval()
    local_rank = distributed.get_rank()

    metric_dict = {str(threshold): 0.0 for threshold in recall_thresholds}
    metric_dict["gt_num"] = 0
    det_annos = []

    for pc_batch in tqdm(val_loader, desc="Step", disable=local_rank != 0, dynamic_ncols=True):
        pc_batch: PCBatch = pc_batch.to(device)
        ret: ModelOutput = model(pc_batch)

        for threshold, recall in ret.recall_dict.items():
            metric_dict[threshold] += recall
        metric_dict["gt_num"] += ret.gt

        det_annos += val_set.gen_pred_dicts(pc_batch, ret.preds)

    # TODO better typing (also rank condition)
    det_annos = merge_dist_data(det_annos, len(val_set))
    metric_dict = merge_dist_data(metric_dict, distributed.get_world_size())

    if local_rank != 0:
        return {}

    ret_dict = {}
    gt_num_cnt = metric_dict["gt_num"]
    for threshold in recall_thresholds:
        threshold_str = str(threshold)
        cur_roi_recall = metric_dict[threshold_str] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric_dict[threshold_str] / max(gt_num_cnt, 1)
        logger.info("Recall ROI %s: %f", threshold, cur_roi_recall)
        logger.info("Recall RCNN %s: %f", threshold, cur_rcnn_recall)
        ret_dict[f"recall/roi_{threshold}"] = cur_roi_recall
        ret_dict[f"recall/rcnn_{threshold}"] = cur_rcnn_recall

    total_pred_objects: int = sum([len(anno["name"]) for anno in det_annos])
    logger.info(
        "Average predicted number of objects (%d samples): %.3f",
        len(val_set),
        total_pred_objects / max(1, len(val_set)),
    )

    with ckpt_path.with_name(f"det_annos_{ckpt_path.stem}.pickle").open("wb") as det_file:
        pickle.dump(det_annos, det_file)

    result_dict = get_evaluation_result(det_annos, val_set)
    with ckpt_path.with_name(f"ret_dict_{ckpt_path.stem}.pickle").open("wb") as det_file:
        pickle.dump(result_dict, det_file)


def get_evaluation_result(det_annos, val_set: IDataset):
    result_str, result_dict = val_set.evaluation(det_annos)
    logger.info(result_str)
    return result_dict


def initialize_data(data_cfg: DictConfig, batch_size: int, num_workers: int):
    val_set: IDataset = hydra.utils.instantiate(data_cfg, split="val")
    val_sampler = FlatDistSampler(val_set)
    val_loader = DataLoader(
        val_set,
        batch_size,
        collate_fn=val_set.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    return val_loader, val_set


def load_checkpoint(model: nn.parallel.DistributedDataParallel, path: Path, device: torch.device):
    logger.info("Loading checkpoint %s", path)
    state_dict = torch.load(path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint %s", path)


def merge_dist_data(data, data_size: int):
    # [rank1: [a, b, c], rank2: [d, e, f]]
    # -> [a, d, b, e, c, f]
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    tmp_dir = Path(tempfile.gettempdir())

    distributed.barrier()
    part_path = tmp_dir / f"result_part_{local_rank}.pickle"
    with part_path.open("wb") as part_file:
        pickle.dump(data, part_file)
    distributed.barrier()

    if local_rank != 0:
        return

    if isinstance(data, list):
        part_list = []
        for rank in range(world_size):
            part_path = tmp_dir / f"result_part_{rank}.pickle"
            with part_path.open("rb") as part_file:
                part_list.append(pickle.load(part_file))
        ordered_results = list(itertools.chain.from_iterable(zip(*part_list)))
        return ordered_results[:data_size]
    elif isinstance(data, dict):
        part_path = tmp_dir / "result_part_0.pickle"
        with part_path.open("rb") as part_file:
            sum_dict: Dict[str, float] = pickle.load(part_file)

        for rank in range(world_size):
            part_path = tmp_dir / f"result_part_{rank}.pickle"
            with part_path.open("rb") as part_file:
                data_dict = pickle.load(part_file)
            for key in sum_dict:
                sum_dict[key] += data_dict[key]
        return sum_dict
    else:
        raise NotImplementedError


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_grid_size", resolver.get_grid_size)
    OmegaConf.register_new_resolver("len", lambda arr: len(arr))
    main()
