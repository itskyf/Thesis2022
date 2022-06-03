from dataclasses import dataclass
from typing import Optional

import torch

from ...ops.iou3d_nms import iou3d_nms_utils


@dataclass
class NMSConf:
    ...
    name: str
    multi_classes_nms: bool  # TODO only SECOND multi use this variable
    pre_maxsize: int
    post_maxsize: int
    threshold: float
    score_thresh: Optional[float] = None


def class_agnostic_nms(box_scores: torch.Tensor, box_preds: torch.Tensor, nms_cfg: NMSConf):
    scores_mask = None
    src_box_scores = box_scores
    # TODO print this value in colab
    if nms_cfg.score_thresh is not None:
        scores_mask = box_scores >= nms_cfg.score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(
            box_scores, k=min(nms_cfg.pre_maxsize, box_scores.shape[0])
        )
        boxes_for_nms = box_preds[indices]
        keep_idx = getattr(iou3d_nms_utils, nms_cfg.name)(
            boxes_for_nms[:, 0:7], box_scores_nms, nms_cfg.threshold
        )[0]
        selected = indices[keep_idx[: nms_cfg.post_maxsize]]

    if scores_mask is not None:
        original_idxs = scores_mask.nonzero().flatten()
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores: torch.Tensor, box_preds: torch.Tensor, nms_cfg: NMSConf):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.size(1)):
        if nms_cfg.score_thresh is not None:
            scores_mask = cls_scores[:, k] >= nms_cfg.score_thresh
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(
                box_scores, k=min(nms_cfg.pre_maxsize, box_scores.size(0))
            )
            boxes_for_nms = cur_box_preds[indices]
            keep_idx = getattr(iou3d_nms_utils, nms_cfg.name)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_cfg.threshold
            )[0]
            selected = indices[keep_idx[: nms_cfg.post_maxsize]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores)
    pred_labels = torch.cat(pred_labels)
    pred_boxes = torch.cat(pred_boxes)
    return pred_scores, pred_labels, pred_boxes
