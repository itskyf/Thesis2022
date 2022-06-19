import logging
import pickle
import time
from pathlib import Path

import torch
import tqdm
from torch import distributed, nn
from torch.utils.data import DataLoader

from pcdet.datasets import DatasetTemplate
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f"recall_roi_{cur_thresh}"] += ret_dict.get(f"roi_{cur_thresh}", 0)
        metric[f"recall_rcnn_{cur_thresh}"] += ret_dict.get(f"rcnn_{cur_thresh}", 0)
    metric["gt_num"] += ret_dict.get("gt", 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]

    recall_roi = metric[f"recall_roi_{min_thresh}"]
    recall_rcnn = metric[f"recall_rcnn_{min_thresh}"]
    disp_dict[f"recall_{min_thresh}"] = f"({recall_roi}, {recall_rcnn}) / {metric['gt_num']}"


@torch.no_grad()
def eval_one_epoch(
    cfg,
    model: nn.parallel.DistributedDataParallel,
    dataloader: DataLoader,
    local_rank: int,
    logger: logging.Logger,
    eval_dir: Path,
):
    model.eval()
    metric = {"gt_num": 0}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f"recall_roi_{cur_thresh}"] = 0
        metric[f"recall_rcnn_{cur_thresh}"] = 0

    dataset: DatasetTemplate = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    progress_bar = (
        tqdm.tqdm(total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True)
        if local_rank == 0
        else None
    )
    start_time = time.time()

    for _, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        det_annos += dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names)
        if progress_bar is not None:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()

    tmpdir = eval_dir / "tmpdir"
    world_size = distributed.get_world_size()
    det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir)
    metric = common_utils.merge_results_dist([metric], world_size, tmpdir)

    sec_per_example = (time.time() - start_time) / len(dataset)
    logger.info("Generate label finished (%.4f seconds per sample)", sec_per_example)

    if local_rank != 0:
        return {}

    for key in metric[0]:
        for k in range(1, world_size):
            metric[0][key] += metric[k][key]
    metric = metric[0]

    gt_num_cnt = metric["gt_num"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric[f"recall_roi_{cur_thresh}"] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric[f"recall_rcnn_{cur_thresh}"] / max(gt_num_cnt, 1)
        logger.info("recall_roi_%s: %f", cur_thresh, cur_roi_recall)
        logger.info("recall_rcnn_%s: %f", cur_thresh, cur_rcnn_recall)

    total_pred_objects = sum(len(anno["name"]) for anno in det_annos)

    logger.info(
        "Average predicted number of objects (%d samples): %.3f",
        len(det_annos),
        total_pred_objects / max(1, len(det_annos)),
    )

    with (eval_dir / "det_annos.pkl").open("wb") as det_annos_file:
        pickle.dump(det_annos, det_annos_file)
    logger.info("Predictions is save to %s", eval_dir)
    return dataset.evaluation(det_annos, class_names)
