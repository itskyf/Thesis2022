from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from ...utils import common_utils, loss_utils
from ...utils.box_coder_utils import IBoxCoder
from ..model_utils.model_nms_utils import NMSConf, class_agnostic_nms
from .target_assigner import ProposalTargetLayer, RoIData


@dataclass
class LossWeights:
    rcnn_classification: float
    rcnn_regression: float
    rcnn_corner: float


class IRoIHead(nn.Module):
    def __init__(
        self,
        box_coder: IBoxCoder,
        proposal_target_layer: ProposalTargetLayer,
        corner_loss_regularization: bool,
        classification_loss_fn: nn.Module,
        reg_loss_fn: nn.Module,
        loss_weights: LossWeights,
    ):
        super().__init__()
        self.box_coder = box_coder
        self.proposal_target_layer = proposal_target_layer

        self.corner_loss_regularization = corner_loss_regularization
        self.classification_loss_fn = classification_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.loss_weights = loss_weights
        self.fw_data: RoIData

    def assign_targets(
        self,
        gt_boxes: torch.Tensor,
        rois: torch.Tensor,
        roi_labels: torch.Tensor,
        roi_scores: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.fw_data: RoIData = self.proposal_target_layer(
                gt_boxes, rois, roi_labels, roi_scores, batch_size
            )

        # canonical transformation
        rois = self.fw_data.rois
        gt_of_rois = self.fw_data.gt_of_rois

        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.flatten()
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi / 2) & (heading_label < np.pi * 1.5)
        # (0 ~ pi/2, 3pi/2 ~ 2pi)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        self.fw_data.gt_of_rois = gt_of_rois
        return self.fw_data.rois, self.fw_data.roi_labels

    def get_loss(self):
        rcnn_cls_loss = self.get_box_classification_loss()
        rcnn_corner_loss, rcnn_reg_loss = self.get_box_regression_loss()
        return rcnn_cls_loss, rcnn_corner_loss, rcnn_reg_loss

    def get_box_classification_loss(self):
        rcnn_cls = self.fw_data.rcnn_cls
        rcnn_cls_labels = self.fw_data.rcnn_cls_labels.flatten()
        if isinstance(self.classification_loss_fn, nn.BCELoss):
            rcnn_cls = torch.sigmoid(rcnn_cls.flatten())
            rcnn_cls_labels = rcnn_cls_labels.float()
        batch_loss_cls = self.classification_loss_fn(rcnn_cls, rcnn_cls_labels)
        cls_valid_mask = (rcnn_cls_labels >= 0).float()
        rcnn_cls_loss = torch.sum(batch_loss_cls * cls_valid_mask)
        rcnn_cls_loss /= torch.clamp(cls_valid_mask.sum(), min=1.0)
        return rcnn_cls_loss * self.loss_weights.rcnn_classification

    def get_box_regression_loss(self):
        code_size = self.box_coder.code_size
        reg_valid_mask = self.fw_data.reg_valid_mask.flatten()
        gt_boxes3d_ct = self.fw_data.gt_of_rois[..., 0:code_size]
        gt_of_rois_src = self.fw_data.gt_of_rois_src[..., 0:code_size].view(-1, code_size)
        rcnn_reg = self.fw_data.rcnn_reg  # (rcnn_batch_size, C)
        roi_boxes3d = self.fw_data.rois
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()

        rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0
        reg_targets = self.box_coder.encode_torch(
            gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
        )

        # [B, M, 7]
        rcnn_reg_loss = self.reg_loss_fn(
            rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0), reg_targets.unsqueeze(dim=0)
        )
        rcnn_reg_loss = rcnn_reg_loss.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()
        rcnn_reg_loss = rcnn_reg_loss.sum() / max(fg_sum, 1) * self.loss_weights.rcnn_regression

        rcnn_corner_loss = None
        if self.corner_loss_regularization and fg_sum > 0:
            # TODO: NEED to BE CHECK
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
            batch_anchors = fg_roi_boxes3d.clone().detach()
            roi_ry = fg_roi_boxes3d[:, :, 6].flatten()
            roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
            batch_anchors[:, :, 0:3] = 0
            rcnn_boxes3d = self.box_coder.decode_torch(
                fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
            ).view(-1, code_size)

            rcnn_boxes3d = common_utils.rotate_points_along_z(
                rcnn_boxes3d.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)
            rcnn_boxes3d[:, 0:3] += roi_xyz

            rcnn_corner_loss = loss_utils.get_corner_loss_lidar(
                rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7]
            )
            rcnn_corner_loss = rcnn_corner_loss.mean()
            rcnn_corner_loss = rcnn_corner_loss * self.loss_weights.rcnn_corner

        return rcnn_corner_loss, rcnn_reg_loss

    def generate_predicted_boxes(
        self, batch_size: int, rois: torch.Tensor, cls_preds: torch.Tensor, box_preds: torch.Tensor
    ):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].flatten()
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois)
        batch_box_preds = batch_box_preds.view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_box_preds, batch_cls_preds

    # TODO is it run_nms?
    @staticmethod
    @torch.no_grad()
    def run_nms(
        batch_box_preds: torch.Tensor,
        batch_cls_preds: torch.Tensor,
        batch_size: int,
        nms_cfg: NMSConf,
        batch_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        rois = batch_box_preds.new_zeros(
            (batch_size, nms_cfg.post_maxsize, batch_box_preds.shape[-1])
        )
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_cfg.post_maxsize))
        roi_labels = torch.zeros(
            (batch_size, nms_cfg.post_maxsize), dtype=torch.int64, device=batch_box_preds.device
        )

        for index in range(batch_size):
            if batch_index is not None:
                assert batch_cls_preds.dim() == 2
                batch_mask = batch_index == index
            else:
                assert batch_cls_preds.dim() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_cfg.multi_classes_nms:
                raise NotImplementedError
            selected = class_agnostic_nms(
                box_scores=cur_roi_scores, box_preds=box_preds, nms_cfg=nms_cfg
            )[0]

            rois[index, : len(selected), :] = box_preds[selected]
            roi_scores[index, : len(selected)] = cur_roi_scores[selected]
            roi_labels[index, : len(selected)] = cur_roi_labels[selected]

        roi_labels += 1
        has_class_labels = batch_cls_preds.size(-1) > 1
        # TODO check batch_index
        return rois, roi_labels, roi_scores, has_class_labels
