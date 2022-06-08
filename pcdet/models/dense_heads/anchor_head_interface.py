from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ...utils import common_utils, loss_utils
from ...utils.box_coder_utils import IBoxCoder, PreviousResidualDecoder
from .target_assigner import AnchorConf, AnchorGenerator, ITargetAssigner


@dataclass
class ForwardData:
    box_preds: torch.Tensor = field(init=False)
    cls_labels: torch.Tensor = field(init=False)
    cls_preds: torch.Tensor = field(init=False)
    dir_cls_preds: torch.Tensor = field(init=False)
    reg_targets: torch.Tensor = field(init=False)
    reg_weights: torch.Tensor = field(init=False)


@dataclass
class LossWeight:
    classification: float
    direction: float
    location: float


class IAnchorHead(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        dir_offset: float,
        dir_limit_offset: float,
        loss_weights: LossWeight,
        num_class: int,
        num_dir_bins: int,
        box_coder: IBoxCoder,
        target_assigner_partial,  # TODO typing
        anchor_range: List[int],
        anchor_cfgs: List[AnchorConf],
        use_direction_classifier: bool,
        grid_size: npt.NDArray[np.int32],
        reg_loss: nn.Module,
        use_multihead: bool = False,
    ):
        super().__init__()
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.loss_weights = loss_weights
        self.num_class = num_class
        self.num_dir_bins = num_dir_bins
        self.use_multihead = use_multihead

        self.box_coder = box_coder
        self.target_assigner: ITargetAssigner = target_assigner_partial(
            box_coder=box_coder, use_multihead=use_multihead
        )

        anchors, num_anchors_per_loc_list = _generate_anchors(
            anchor_range, anchor_cfgs, grid_size, anchor_ndim=self.box_coder.code_size
        )
        for i, anchor in enumerate(anchors):
            self.register_buffer(f"anchor_{i}", anchor)
        self.anchors_len = len(anchors)
        self.num_anchors_per_location = sum(num_anchors_per_loc_list)

        # Use in derived class
        self.conv_dir_cls = (
            nn.Conv2d(in_channels, self.num_anchors_per_location * num_dir_bins, kernel_size=1)
            if use_direction_classifier
            else None
        )

        self.cls_loss_fn = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.dir_loss_fn = loss_utils.WeightedCrossEntropyLoss()
        self.reg_loss_fn = reg_loss
        self.fw_data = ForwardData()

    def forward(self, spatial_features_2d: torch.Tensor, gt_boxes: torch.Tensor, batch_size: int):
        if self.training:
            cls_labels, reg_targets, reg_weights = self.target_assigner.assign_targets(
                self._get_anchors(), gt_boxes
            )
            self.fw_data.cls_labels = cls_labels
            self.fw_data.reg_targets = reg_targets
            self.fw_data.reg_weights = reg_weights
        return self._forward_impl(spatial_features_2d, batch_size)

    @abstractmethod
    def _forward_impl(self, spatial_features_2d: torch.Tensor, batch_size: int):
        ...

    def _get_anchors(self):
        return [getattr(self, f"anchor_{i}") for i in range(self.anchors_len)]

    def generate_predicted_boxes(
        self,
        batch_size: int,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        dir_cls_preds: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        anchors = self._get_anchors()
        anchors = (
            torch.cat(
                [
                    anchor.permute(3, 4, 0, 1, 2, 5).reshape(-1, anchor.shape[-1])
                    for anchor in anchors
                ]
            )
            if self.use_multihead
            else torch.cat(anchors, dim=-3)
        )
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
        batch_box_preds = (
            box_preds.view(batch_size, num_anchors, -1)
            if not isinstance(box_preds, list)
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        )
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = (
                dir_cls_preds.view(batch_size, num_anchors, -1)
                if not isinstance(dir_cls_preds, list)
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            )
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = 2 * np.pi / self.num_dir_bins
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - self.dir_offset, self.dir_limit_offset, period
            )
            batch_box_preds[..., 6] = (
                dir_rot + self.dir_offset + period * dir_labels.to(batch_box_preds.dtype)
            )

        if isinstance(self.box_coder, PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def get_loss(self):
        cls_loss = self.get_classification_loss()
        dir_loss, loc_loss = self.get_box_regression_loss()
        return cls_loss, dir_loss, loc_loss

    def get_classification_loss(self):
        cls_preds = self.fw_data.cls_preds
        box_cls_labels = self.fw_data.cls_labels
        batch_size = int(cls_preds.shape[0])

        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *cls_targets.shape, self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_fn(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        return cls_loss * self.loss_weights.classification

    def get_box_regression_loss(self):
        box_preds = self.fw_data.box_preds
        box_dir_cls_preds = self.fw_data.dir_cls_preds
        box_reg_targets = self.fw_data.reg_targets
        box_cls_labels = self.fw_data.cls_labels
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        anchors = self._get_anchors()
        anchors = (
            torch.cat(
                [
                    anchor.permute(3, 4, 0, 1, 2, 5).reshape(-1, anchor.shape[-1])
                    for anchor in anchors
                ]
            )
            if self.use_multihead
            else torch.cat(anchors, dim=-3)
        )
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(
            batch_size,
            -1,
            box_preds.shape[-1] // self.num_anchors_per_location
            if not self.use_multihead
            else box_preds.shape[-1],
        )
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = _add_sin_difference(box_preds, box_reg_targets)
        # [N, M]
        loc_loss_src = self.reg_loss_fn(box_preds_sin, reg_targets_sin, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.loss_weights.location

        dir_loss = None
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(anchors, box_reg_targets)
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.num_dir_bins)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_fn(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.loss_weights.direction

        return dir_loss, loc_loss

    def get_direction_target(self, anchors: torch.Tensor, reg_targets: torch.Tensor):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - self.dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / self.num_dir_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=self.num_dir_bins - 1)

        # One-hot encoding
        dir_targets = torch.zeros(
            *dir_cls_targets.shape,
            self.num_dir_bins,
            dtype=anchors.dtype,
            device=dir_cls_targets.device,
        )
        dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        return dir_targets


def _add_sin_difference(boxes1: torch.Tensor, boxes2: torch.Tensor, dim: int = 6):
    assert dim != -1
    rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
        boxes2[..., dim : dim + 1]
    )
    rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(boxes2[..., dim : dim + 1])
    boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1)
    return boxes1, boxes2


def _generate_anchors(
    anchor_range: List[int],
    cfgs: List[AnchorConf],
    grid_size: npt.NDArray[np.int32],
    anchor_ndim: int = 7,
):
    generator = AnchorGenerator(anchor_range, cfgs)
    feature_map_size = np.array([grid_size[:2] // cfg.feature_map_stride for cfg in cfgs])
    anchors_list, num_anchors_per_loc_list = generator.generate_anchors(feature_map_size)
    if anchor_ndim != 7:
        for idx, anchors in enumerate(anchors_list):
            pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
            new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
            anchors_list[idx] = new_anchors

    return anchors_list, num_anchors_per_loc_list
