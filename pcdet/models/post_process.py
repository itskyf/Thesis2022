from collections import defaultdict
from itertools import zip_longest
from typing import NamedTuple, Optional

import torch

from ..ops.iou3d_nms import iou3d_nms_utils
from .nms import class_agnostic_nms


class BatchPrediction(NamedTuple):
    pred_dicts: list[dict[str, torch.Tensor]]
    total_gt: int
    recall_dict: dict[str, float]


def post_processing(
    batch_cls_preds: torch.Tensor,
    batch_box_preds: torch.Tensor,
    batch_gt_boxes: Optional[torch.Tensor],
    num_class: int,
    nms_cfg,
    post_cfg,
):
    """
    Args:
        b_cls_preds [B, last_n_point, 3]
        b_box_preds [B, last_n_point, 7]
    Returns:
    """
    l_gt_boxes = batch_gt_boxes if batch_gt_boxes is not None else []
    pred_dicts = []
    total_gt = 0
    recall_dict = defaultdict(float)
    for cls_preds, box_preds, gt_boxes in zip_longest(batch_cls_preds, batch_box_preds, l_gt_boxes):
        assert cls_preds.size(1) in (1, num_class)
        cls_preds = torch.sigmoid(cls_preds)
        # TODO multi-classes nms
        box_scores, label_preds = torch.max(cls_preds, dim=-1)
        label_preds += 1

        selected, selected_scores = class_agnostic_nms(
            box_scores,
            box_preds,
            post_cfg.score_thresh,
            nms_cfg.type,
            nms_cfg.threshold,
            nms_cfg.pre_maxsize,
            nms_cfg.post_maxsize,
        )
        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = box_preds[selected]

        if post_cfg.recall_mode == "normal" and gt_boxes is not None:
            num_gt, b_recall_dict = _generate_recall_record(
                final_boxes, gt_boxes, post_cfg.thresh_list
            )
            total_gt += num_gt
            for key, val in b_recall_dict.items():
                recall_dict[key] += val

        record_dict = {
            "pred_boxes": final_boxes,
            "pred_scores": final_scores,
            "pred_labels": final_labels,
        }
        pred_dicts.append(record_dict)

    return BatchPrediction(pred_dicts, total_gt, recall_dict)


def _generate_recall_record(
    box_preds: torch.Tensor, b_gt_boxes: torch.Tensor, thresh_list: list[int]
):
    total_gt = 0
    recall_dict = defaultdict(float)

    # gt_boxes is padded with zeros, find its true len
    k = len(b_gt_boxes) - 1
    while k >= 0 and torch.sum(b_gt_boxes[k]) == 0:
        k -= 1
    b_gt_boxes = b_gt_boxes[: k + 1]

    if (num_gt := b_gt_boxes.size(0)) > 0:
        total_gt += num_gt
        iou3d_rcnn = (
            iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, :7], b_gt_boxes[:, :7])
            if box_preds.size(0) > 0
            else torch.zeros((0, num_gt))
        )
        for cur_thresh in thresh_list:
            if iou3d_rcnn.size(0) > 0:
                rcnn_recalled = torch.sum(iou3d_rcnn.max(dim=0).values > cur_thresh)
                recall_dict[f"rcnn_{cur_thresh}"] += rcnn_recalled.item()
    return total_gt, recall_dict
