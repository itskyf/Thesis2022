from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...datasets import Prediction
from ...ops.iou3d_nms import iou3d_nms_utils
from ..model_utils.model_nms_utils import NMSConf, class_agnostic_nms, multi_classes_nms


@dataclass
class ModelOutput:
    gt: int
    preds: List[Prediction]
    recall_dict: Dict[str, float]


class IDetector3D(nn.Module):
    def __init__(
        self,
        num_class: int,
        nms_cfgs: NMSConf,
        output_raw_score: bool,
        recall_thresholds: List[float],
    ):
        super().__init__()
        self.num_class = num_class
        self.nms_cfgs = nms_cfgs
        self.output_raw_score = output_raw_score
        self.recall_thresholds = recall_thresholds

    def post_processing(
        self,
        batch_box_preds: torch.Tensor,
        batch_cls_preds: Union[torch.Tensor, List[torch.Tensor]],
        gt_boxes: torch.Tensor,
        rois: torch.Tensor,  # TODO perhaps this variable is optional
        roi_labels: torch.Tensor,
        batch_size: int,
        has_class_labels: bool,
        batch_index: Optional[torch.Tensor] = None,
        multihead_label_mapping: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        num_gt = 0
        preds: List[Prediction] = []
        recall_dict = {str(recall_threshold): 0.0 for recall_threshold in self.recall_thresholds}
        for index in range(batch_size):
            if batch_index is not None:
                assert batch_box_preds.dim() == 2
                batch_mask = batch_index == index
            else:
                assert batch_box_preds.dim() == 3
                batch_mask = index

            box_preds = batch_box_preds[batch_mask]
            src_box_preds = box_preds

            # TODO better condition (right now only SECOND multihead use list branch)
            if not isinstance(batch_cls_preds, list):
                cls_preds = batch_cls_preds[batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.size(1) in [1, self.num_class]
                cls_preds = torch.sigmoid(cls_preds)

                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                label_preds = roi_labels[index] if has_class_labels else label_preds + 1
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds, nms_cfg=self.nms_cfgs
                )

                if self.output_raw_score:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            else:
                # AnchorHeadMulti
                cls_preds = [x[batch_mask] for x in batch_cls_preds]
                src_cls_preds = cls_preds  # TODO necessary?
                cls_preds = [torch.sigmoid(x) for x in cls_preds]

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                assert multihead_label_mapping is not None
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.size(1) == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx : cur_start_idx + cur_cls_preds.size(0)]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_cfg=self.nms_cfgs
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.size(0)

                final_scores = torch.cat(pred_scores)
                final_labels = torch.cat(pred_labels)
                final_boxes = torch.cat(pred_boxes)

            sample_recall_dict, gt = _generate_sample_recall(
                box_preds=src_box_preds,  # TODO original also use final_boxes
                idx_gt_boxes=gt_boxes[index],
                idx_rois=rois[index],
                thresh_list=self.recall_thresholds,
            )
            num_gt += gt
            for threshold, recall in sample_recall_dict.items():
                recall_dict[threshold] += recall
            preds.append(Prediction(final_boxes, final_labels, final_scores))

        return ModelOutput(num_gt, preds, recall_dict)


def _generate_sample_recall(
    box_preds: torch.Tensor,
    idx_gt_boxes: torch.Tensor,
    idx_rois: torch.Tensor,
    thresh_list: List[float],
) -> Tuple[Dict[str, float], int]:
    k = len(idx_gt_boxes) - 1
    while k >= 0 and idx_gt_boxes[k].sum() == 0:
        k -= 1
    idx_gt_boxes = idx_gt_boxes[: k + 1]

    recall_dict = defaultdict(float)
    gt = idx_gt_boxes.size(0)
    if gt > 0:
        iou3d_rcnn = (
            iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], idx_gt_boxes[:, 0:7])
            if box_preds.size(0) > 0
            else torch.zeros((0, idx_gt_boxes.size(0)))
        )
        if idx_rois is not None:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(idx_rois[:, 0:7], idx_gt_boxes[:, 0:7])

        for cur_thresh in thresh_list:
            thresh_str = str(cur_thresh)
            if iou3d_rcnn.size(0) == 0:
                recall_dict[thresh_str] += 0  # TODO ???
            else:
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict[thresh_str] += rcnn_recalled
            if idx_rois is not None:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict[thresh_str] += roi_recalled

    return recall_dict, gt
