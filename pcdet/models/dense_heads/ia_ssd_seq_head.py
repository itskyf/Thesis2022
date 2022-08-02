import enum
from typing import NamedTuple, Optional

import torch
from torch import nn

from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ...utils import box_utils
from ...utils.box_coder_utils import PointResidualBinOriCoder


class Target(NamedTuple):
    pt_cls_labels: torch.Tensor  # B, N
    box_idxs_labels: torch.Tensor  # B, N
    gt_box_of_points: torch.Tensor  # B, N, 8
    gt_box_of_fg_pts: torch.Tensor  # NFG, 8
    pt_box_labels: Optional[torch.Tensor]  # B, N, 8


class Targets(NamedTuple):
    center: Target
    ctr_origin: Target
    sa_ins: list[Target]


class TargetMode(enum.Enum):
    EXT_GT = enum.auto()
    IGNORE_FLAG = enum.auto()


class IASSDSeqHead(nn.Module):
    """
    A simple point-based detect head, which are used for IA-SSD.
    """

    def __init__(
        self,
        n_class: int,
        in_channels: int,
        mid_channels: int,
        bin_size: int,
        mean_size: list[list[int]],
        gt_ext_dims: list[float],
        org_ext_dims: list[float],
    ):
        _mean_size = torch.tensor(mean_size)
        assert _mean_size.shape == (3, 3)
        assert len(gt_ext_dims) == 3
        assert len(org_ext_dims) == 3
        super().__init__()
        # xyz, o, whl, c

        self.box_coder = PointResidualBinOriCoder(bin_size)
        self.xyz_conv = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )
        self.xyz_head = nn.Linear(mid_channels, 3)

        in_channels += mid_channels
        ori_size = self.box_coder.code_size - 6
        self.ori_conv = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )
        self.ori_head = nn.Linear(mid_channels, ori_size)

        self.whl_conv = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )
        self.whl_head = nn.Linear(mid_channels, 3)

        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, n_class),
        )

        self.register_buffer("mean_size", _mean_size, persistent=False)
        self.gt_ext_dims = gt_ext_dims
        self.org_ext_dims = org_ext_dims

    def forward(
        self,
        ctr_feats: torch.Tensor,
        ctr_preds: torch.Tensor,
        ctr_origins: torch.Tensor,
        pts_list: list[torch.Tensor],
        gt_boxes: Optional[torch.Tensor],
    ):
        """
        Args:
            ctr_feats [B, C, last_n_point]
            ctr_preds, ctr_origins [B, last_n_point, 3]
            gt_boxes: [B, ?, 8]
        Returns:
            ctr_cls_preds [B, last_n_point, 3]
            ctr_box_preds [B, last_n_point, box_coder.code_size]
            pt_box_preds [B, last_n_point, 7]
        """
        ctr_feats = ctr_feats.transpose(1, 2)

        xyz_feat = self.xyz_conv(ctr_feats)
        xyz = self.xyz_head(xyz_feat)

        ori_feat = torch.cat([ctr_feats, xyz_feat], dim=-1)
        ori_feat = self.ori_conv(ori_feat)
        ori = self.ori_head(ori_feat)

        whl_feat = torch.cat([ctr_feats, ori_feat], dim=-1)
        whl_feat = self.whl_conv(whl_feat)
        whl = self.whl_head(whl_feat)

        cls_feat = torch.cat([ctr_feats, whl_feat], dim=-1)

        ctr_box_preds = torch.cat([xyz, whl, ori], dim=-1)
        ctr_cls_preds = self.cls_head(cls_feat)
        pred_classes = ctr_cls_preds.max(dim=-1).indices
        pt_box_preds = self.box_coder.decode_torch(
            ctr_box_preds, ctr_preds, pred_classes + 1, self.get_buffer("mean_size")
        )

        targets = (
            self.assign_targets(ctr_preds, ctr_origins, pts_list, gt_boxes)
            if gt_boxes is not None
            else None
        )
        return ctr_cls_preds, ctr_box_preds, pt_box_preds, targets

    def assign_stack_targets(
        self,
        points_batch: torch.Tensor,
        gt_boxes_batch: torch.Tensor,
        extra_width: list[float],
        mode: TargetMode,
        ret_box_labels=False,
    ):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        points_batch = points_batch.detach()
        total_pts = points_batch.size(1)

        ext_gt_boxes_b = box_utils.enlarge_box3d(gt_boxes_batch, extra_width)
        point_cls_labels_list = []
        box_idxs_labels_list = []
        gt_box_of_points_list = []
        gt_boxes_of_fg_points_list = []
        point_box_labels_list = [] if ret_box_labels else None

        for gt_boxes, ext_gt_boxes, points in zip(gt_boxes_batch, ext_gt_boxes_b, points_batch):
            u_points = points.unsqueeze(dim=0)

            box_idxs_of_pts = points_in_boxes_gpu(u_points, gt_boxes[None, :, :7])
            box_idxs_of_pts = box_idxs_of_pts.squeeze(dim=0)
            box_fg_flag = box_idxs_of_pts >= 0

            ext_box_idxs_of_pts = points_in_boxes_gpu(u_points, ext_gt_boxes[None, :, :7])
            ext_box_idxs_of_pts = ext_box_idxs_of_pts.squeeze(dim=0)
            ext_fg_flag = ext_box_idxs_of_pts >= 0

            point_cls_labels = torch.zeros_like(box_idxs_of_pts)
            if mode is TargetMode.EXT_GT:
                ext_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag]
                # instance points should keep unchanged
                fg_flag = ext_fg_flag
                box_idxs_of_pts = ext_box_idxs_of_pts
            elif mode is TargetMode.IGNORE_FLAG:
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ ext_fg_flag
                point_cls_labels[ignore_flag] = -1
            else:
                raise NotImplementedError

            box_idxs_labels_list.append(box_idxs_of_pts)
            gt_box_of_fg_points = gt_boxes[box_idxs_of_pts[fg_flag]]
            point_cls_labels[fg_flag] = gt_box_of_fg_points[..., -1].long()
            point_cls_labels_list.append(point_cls_labels)
            bg_flag = point_cls_labels == 0  # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[box_idxs_of_pts[fg_flag]]
            gt_boxes_of_fg_points_list.append(gt_box_of_fg_points)
            gt_box_of_points_list.append(gt_boxes[box_idxs_of_pts])

            if point_box_labels_list is not None:
                point_box_labels = points.new_zeros(total_pts, 8)
                if gt_box_of_fg_points.size(0) > 0:
                    point_box_labels[fg_flag] = self.box_coder.encode_torch(
                        gt_box_of_fg_points[:, :-1],
                        points[fg_flag],
                        gt_box_of_fg_points[:, -1].long(),
                        self.get_buffer("mean_size"),
                    )
                point_box_labels_list.append(point_box_labels)

        point_cls_labels = torch.stack(point_cls_labels_list)
        box_idxs_labels = torch.stack(box_idxs_labels_list)
        gt_box_of_points = torch.stack(gt_box_of_points_list)
        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points_list)
        point_box_labels = (
            torch.stack(point_box_labels_list) if point_box_labels_list is not None else None
        )
        return Target(
            point_cls_labels,
            box_idxs_labels,
            gt_box_of_points,
            gt_boxes_of_fg_points,
            point_box_labels,
        )

    def assign_targets(
        self,
        ctr_preds: torch.Tensor,
        ctr_origins: torch.Tensor,
        pts_list: list[torch.Tensor],
        gt_boxes: torch.Tensor,
    ):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        if gt_boxes.size(-1) == 10:  # nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        center_targets = self.assign_stack_targets(
            ctr_preds, gt_boxes, self.gt_ext_dims, TargetMode.IGNORE_FLAG, ret_box_labels=True
        )

        org_targets = self.assign_stack_targets(
            ctr_origins, gt_boxes, self.org_ext_dims, TargetMode.EXT_GT, ret_box_labels=True
        )

        sa_targets = [
            self.assign_stack_targets(
                pts,
                gt_boxes,
                [0.5, 0.5, 0.5],
                TargetMode.IGNORE_FLAG if i == 0 else TargetMode.EXT_GT,
            )
            for i, pts in enumerate(pts_list)
        ]

        return Targets(center_targets, org_targets, sa_targets)
