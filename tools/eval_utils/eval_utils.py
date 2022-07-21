from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcdet.datasets import DatasetTemplate, load_data_to_gpu
from pcdet.models.post_process import BatchPrediction


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    val_set: DatasetTemplate,
    val_loader: DataLoader,
    thresh_list: list[float],
    n_pts: tuple[int, ...],
):
    total_gt = 0
    recall_dict = defaultdict(float)
    cls_total_gt = [0] * 3  # TODO remove hardcore 3
    ins_recall_dict = defaultdict(float)

    det_annos = []
    model.eval()
    for pcd_dict in tqdm(val_loader, dynamic_ncols=True):
        pcd_dict = load_data_to_gpu(pcd_dict)
        ret: BatchPrediction = model(pcd_dict)
        for key, val in ret.recall_dict.items():
            recall_dict[key] += val
        total_gt += ret.total_gt

        for i in range(3):
            cls_total_gt[i] += ret.cls_total_gt[i]
        for key, val in ret.ins_recall_dict.items():
            ins_recall_dict[key] += val
        det_annos += val_set.generate_prediction_dicts(pcd_dict, ret.pred_dicts)

    num_gt = max(total_gt, 1)
    for key in recall_dict:
        recall_dict[key] /= num_gt
    for key in ins_recall_dict:
        ins_recall_dict[key] /= cls_total_gt[int(key[-1]) - 1]

    total_pred_objects = sum(len(anno["name"]) for anno in det_annos)
    avg_preds = total_pred_objects / max(1, len(det_annos))
    print(f"Average: {avg_preds:.3f} objects over {len(det_annos)} samples")
    return det_annos, recall_dict, ins_recall_dict
