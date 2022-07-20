import torch

from ..ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(
    box_scores: torch.Tensor,
    box_preds: torch.Tensor,
    score_thresh: float,
    nms_type: str,
    nms_thresh: float,
    pre_maxsize: int,
    post_maxsize: int,
):
    """
    Args:
        box_scores [last_n_point]
        box_preds [last_n_point, 7]
    Returns:
    """
    src_box_scores = box_scores
    scores_mask = box_scores >= score_thresh
    box_scores = box_scores[scores_mask]
    box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.size(0) > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(pre_maxsize, box_scores.size(0)))
        boxes_for_nms = box_preds[indices]
        keep_idx = getattr(iou3d_nms_utils, nms_type)(
            boxes_for_nms[:, :7], box_scores_nms, nms_thresh
        )
        selected = indices[keep_idx[:post_maxsize]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().flatten()
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]
