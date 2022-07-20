import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pcdet.datasets import DatasetTemplate, load_data_to_gpu


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    val_set: DatasetTemplate,
    val_loader: DataLoader,
    thresh_list: list[float],
):
    recall_dict = {}
    total_gt = 0
    for cur_thresh in thresh_list:
        recall_dict[f"recall_roi_{cur_thresh}"] = 0
        recall_dict[f"recall_rcnn_{cur_thresh}"] = 0

    det_annos = []
    model.eval()
    for pcd_dict in tqdm(val_loader, dynamic_ncols=True):
        pcd_dict = load_data_to_gpu(pcd_dict)
        batch_gt, pred_dicts, b_recall_dict = model(pcd_dict)
        for cur_thresh in thresh_list:
            recall_dict[f"recall_roi_{cur_thresh}"] += b_recall_dict.get(f"roi_{cur_thresh}", 0)
            recall_dict[f"recall_rcnn_{cur_thresh}"] += b_recall_dict.get(f"rcnn_{cur_thresh}", 0)
        total_gt += batch_gt
        det_annos += val_set.generate_prediction_dicts(pcd_dict, pred_dicts)

    num_gt = max(total_gt, 1)
    for cur_thresh in thresh_list:
        recall_dict[f"recall_roi_{cur_thresh}"] /= num_gt
        recall_dict[f"recall_rcnn_{cur_thresh}"] /= num_gt

    total_pred_objects = sum(len(anno["name"]) for anno in det_annos)
    avg_preds = total_pred_objects / max(1, len(det_annos))
    print(f"Average: {avg_preds:.3f} over {len(det_annos)} samples")
    return det_annos, recall_dict
