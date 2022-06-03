import math
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ...utils.box_coder_utils import IBoxCoder
from .anchor_head_interface import IAnchorHead, LossWeight
from .target_assigner import AnchorConf


@dataclass
class AnchorHeadSingleOut:
    batch_cls_preds: torch.Tensor
    batch_box_preds: torch.Tensor
    cls_preds_normalized: bool


class AnchorHeadSingle(IAnchorHead):
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
        predict_boxes_when_training: bool,
        grid_size: npt.NDArray[np.int32],
        reg_loss: nn.Module,
    ):
        super().__init__(
            in_channels,
            dir_offset,
            dir_limit_offset,
            loss_weights,
            num_class,
            num_dir_bins,
            box_coder,
            target_assigner_partial,  # TODO typing
            anchor_range,
            anchor_cfgs,
            use_direction_classifier,
            grid_size,
            reg_loss,
        )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.conv_cls = nn.Conv2d(
            in_channels, self.num_anchors_per_location * self.num_class, kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            in_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1
        )

        eps = 0.01
        if self.conv_cls.bias is not None:
            nn.init.constant_(self.conv_cls.bias, -math.log((1 - eps) / eps))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward_impl(self, spatial_features_2d: torch.Tensor, batch_size: int):
        # [N, H, W, C]
        cls_preds = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box_preds = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        self.fw_data.cls_preds = cls_preds
        self.fw_data.box_preds = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            self.fw_data.dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size, cls_preds, box_preds, dir_cls_preds
            )
            return AnchorHeadSingleOut(batch_cls_preds, batch_box_preds, False)
