import enum
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ....ops.iou3d_nms import iou3d_nms_utils


@dataclass
class RoIData:
    rois: torch.Tensor
    gt_of_rois: torch.Tensor
    gt_of_rois_src: torch.Tensor = field(init=False)
    gt_iou_of_rois: torch.Tensor
    roi_labels: torch.Tensor
    roi_scores: torch.Tensor
    rcnn_cls_labels: torch.Tensor
    reg_valid_mask: torch.Tensor
    rcnn_cls: torch.Tensor = field(init=False)
    rcnn_reg: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.gt_of_rois_src = self.gt_of_rois.clone().detach()


class ClassScoreType(enum.Enum):
    CLASS = enum.auto()
    ROI_IOU = enum.auto()


class ProposalTargetLayer(nn.Module):
    def __init__(
        self,
        cls_score_type: ClassScoreType,
        cls_bg_thresh: float,
        cls_bg_thresh_lo: float,
        cls_fg_thresh: float,
        fg_ratio: float,
        hard_bg_ratio: float,
        reg_fg_thresh: float,
        roi_per_image: int,
        sample_roi_by_each_class: bool = False,
    ):
        super().__init__()
        self.cls_score_type = cls_score_type
        self.cls_bg_thresh = cls_bg_thresh
        self.cls_bg_thresh_lo = cls_bg_thresh_lo
        self.cls_fg_thresh = cls_fg_thresh
        self.fg_ratio = fg_ratio
        self.hard_bg_ratio = hard_bg_ratio
        self.reg_fg_thresh = reg_fg_thresh
        self.roi_per_image = roi_per_image
        self.sample_roi_by_each_class = sample_roi_by_each_class

    def forward(
        self,
        gt_boxes: torch.Tensor,
        rois: torch.Tensor,
        roi_labels: torch.Tensor,
        roi_scores: torch.Tensor,
        batch_size: int,
    ):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        b_rois, b_gt_of_rois, b_roi_ious, b_roi_labels, b_roi_scores = self.sample_rois_for_rcnn(
            gt_boxes, rois, roi_labels, roi_scores, batch_size
        )
        # regression valid mask
        reg_valid_mask = (b_roi_ious > self.reg_fg_thresh).long()

        # classification label
        if self.cls_score_type == ClassScoreType.CLASS.name:  # TODO convert string to enum
            b_cls_labels = (b_roi_ious > self.cls_fg_thresh).long()
            ignore_mask = (b_roi_ious > self.cls_bg_thresh) & (b_roi_ious < self.cls_fg_thresh)
            b_cls_labels[ignore_mask > 0] = -1
        elif self.cls_score_type == ClassScoreType.ROI_IOU.name:
            bg_mask = b_roi_ious < self.cls_bg_thresh
            fg_mask = b_roi_ious > self.cls_fg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            b_cls_labels = (fg_mask > 0).float()
            sub_bg = b_roi_ious[interval_mask] - self.cls_bg_thresh
            b_cls_labels[interval_mask] = sub_bg / (self.cls_fg_thresh - self.cls_bg_thresh)
        else:
            raise NotImplementedError

        return RoIData(
            rois=b_rois,
            gt_of_rois=b_gt_of_rois,
            gt_iou_of_rois=b_roi_ious,
            roi_labels=b_roi_labels,
            roi_scores=b_roi_scores,
            rcnn_cls_labels=b_cls_labels,
            reg_valid_mask=reg_valid_mask,
        )

    def sample_rois_for_rcnn(
        self,
        gt_boxes: torch.Tensor,
        rois: torch.Tensor,
        roi_labels: torch.Tensor,
        roi_scores: torch.Tensor,
        batch_size: int,
    ):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        # TODO refactor this naming style
        code_size = rois.shape[-1]
        b_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size)
        b_gt_of_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size + 1)
        b_roi_ious = rois.new_zeros(batch_size, self.roi_per_image)
        b_roi_scores = rois.new_zeros(batch_size, self.roi_per_image)
        b_roi_labels = rois.new_zeros((batch_size, self.roi_per_image), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels = rois[index], gt_boxes[index], roi_labels[index]
            k = len(cur_gt) - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[: k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.sample_roi_by_each_class:
                max_overlaps, gt_assignment = _get_max_iou_with_same_class(
                    rois=cur_roi,
                    roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7],
                    gt_labels=cur_gt[:, -1].long(),
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            b_rois[index] = cur_roi[sampled_inds]
            b_roi_labels[index] = cur_roi_labels[sampled_inds]
            b_roi_ious[index] = max_overlaps[sampled_inds]
            b_roi_scores[index] = roi_scores[index][sampled_inds]
            b_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
        return b_rois, b_gt_of_rois, b_roi_ious, b_roi_labels, b_roi_scores

    def subsample_rois(self, max_overlaps: torch.Tensor) -> torch.Tensor:
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = round(self.fg_ratio * self.roi_per_image)
        fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)
        # TODO parentless?
        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().flatten()
        easy_bg_inds = ((max_overlaps < self.cls_bg_thresh_lo)).nonzero().flatten()
        hard_bg_inds = (max_overlaps < self.reg_fg_thresh) & (max_overlaps >= self.cls_bg_thresh_lo)
        hard_bg_inds = hard_bg_inds.nonzero().flatten()

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = (
                torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            )
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
            bg_inds = _sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0]  # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_inds = _sample_bg_inds(
                hard_bg_inds, easy_bg_inds, self.roi_per_image, self.hard_bg_ratio
            )
        else:
            raise NotImplementedError
        return torch.cat((fg_inds, bg_inds))


def _get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    Args:
        rois: (N, 7)
        roi_labels: (N)
        gt_boxes: (N, )
        gt_labels:

    Returns:

    """
    """
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = roi_labels == k
        gt_mask = gt_labels == k
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().flatten()

            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

    return max_overlaps, gt_assignment


def _sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
    if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
        hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
        easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        hard_bg_inds = hard_bg_inds[rand_idx]

        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        easy_bg_inds = easy_bg_inds[rand_idx]

        bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
    elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
        hard_bg_rois_num = bg_rois_per_this_image
        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        bg_inds = hard_bg_inds[rand_idx]
    elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
        easy_bg_rois_num = bg_rois_per_this_image
        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        bg_inds = easy_bg_inds[rand_idx]
    else:
        raise NotImplementedError

    return bg_inds
