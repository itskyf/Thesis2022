from typing import List, Tuple

import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils
from ....utils.box_coder_utils import IBoxCoder
from .anchor_generator import AnchorConf
from .interface import ITargetAssigner


class AxisAlignedTargetAssigner(ITargetAssigner):
    def __init__(
        self,
        anchor_cfgs: List[AnchorConf],
        class_names: List[str],
        match_height: bool,
        norm_by_num_examples: bool,
        pos_fraction: float,
        sample_size: int,
        *,
        box_coder: IBoxCoder,
        use_multihead: bool,
    ):
        self.box_coder = box_coder
        self.class_names = np.array(class_names)
        self.match_height = match_height
        self.norm_by_num_examples = norm_by_num_examples
        self.pos_fraction = pos_fraction
        self.sample_size = sample_size
        self.use_multihead = use_multihead

        self.anchor_class_names = [cfg.class_name for cfg in anchor_cfgs]
        self.matched_thresholds = {cfg.class_name: cfg.matched_threshold for cfg in anchor_cfgs}
        self.unmatched_thresholds = {cfg.class_name: cfg.unmatched_threshold for cfg in anchor_cfgs}

        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(
        self, all_anchors: List[torch.Tensor], gt_boxes_with_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        batch_cls_labels = []
        batch_reg_targets = []
        batch_reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[: cnt + 1]
            cur_gt_classes = gt_classes[k][: cnt + 1].int()

            all_labels = []
            all_reg_targets = []
            all_reg_weights = []
            feature_map_size = None
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                mask = (
                    torch.from_numpy(
                        self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name
                    )
                    if cur_gt_classes.shape[0] > 1
                    else torch.tensor(
                        [self.class_names[c - 1] == anchor_class_name for c in cur_gt_classes],
                        dtype=torch.bool,
                    )
                )
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).reshape(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]

                labels, reg_targets, reg_weights = self.assign_class_targets(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name],
                )
                all_labels.append(labels)
                all_reg_targets.append(reg_targets)
                all_reg_weights.append(reg_weights)

            if feature_map_size is None:
                all_labels = torch.cat([t.flatten() for t in all_labels]).flatten()
                all_reg_targets = torch.cat(
                    [t.view(-1, self.box_coder.code_size) for t in all_reg_targets]
                )
                all_reg_weights = torch.cat([t.flatten() for t in all_reg_weights]).flatten()
            else:
                all_labels = torch.cat(
                    [t.view(*feature_map_size, -1) for t in all_labels], dim=-1
                ).flatten()

                all_reg_targets = torch.cat(
                    [
                        t.view(*feature_map_size, -1, self.box_coder.code_size)
                        for t in all_reg_targets
                    ],
                    dim=-2,
                ).view(-1, self.box_coder.code_size)

                all_reg_weights = torch.cat(
                    [t.view(*feature_map_size, -1) for t in all_reg_weights], dim=-1
                ).flatten()

            batch_cls_labels.append(all_labels)
            batch_reg_targets.append(all_reg_targets)
            batch_reg_weights.append(all_reg_weights)

        batch_cls_labels = torch.stack(batch_cls_labels)
        batch_reg_targets = torch.stack(batch_reg_targets)
        batch_reg_weights = torch.stack(batch_reg_weights)
        return batch_cls_labels, batch_reg_targets, batch_reg_weights

    def assign_class_targets(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        matched_threshold: float = 0.6,
        unmatched_threshold: float = 0.45,
    ):

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.full((num_anchors,), -1, dtype=torch.int32, device=anchors.device)
        gt_ids = torch.full((num_anchors,), -1, dtype=torch.int32, device=anchors.device)

        anchors_with_max_overlap = anchor_to_gt_argmax = gt_inds_force = None
        if len(gt_boxes) > 0 and anchors.size(0) > 0:
            anchor_by_gt_overlap = (
                iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
                if self.match_height
                else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])
            )

            # NOTE: Speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(
            # anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            # gt_to_anchor_argmax = torch.from_numpy(
            # anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)
            ]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = torch.nonzero(labels > 0)[:, 0]

        if self.pos_fraction >= 0:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = torch.nonzero(labels > 0)[:, 0]

            num_bg = self.sample_size - torch.sum(labels > 0)
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if anchors_with_max_overlap is not None and gt_inds_force is not None:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            else:
                labels[:] = 0

        reg_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if anchor_to_gt_argmax is not None:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            reg_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = torch.sum(labels >= 0)
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        return labels, reg_targets, reg_weights
