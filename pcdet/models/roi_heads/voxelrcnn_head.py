import itertools
from dataclasses import dataclass
from typing import List

import spconv.pytorch as spconv
import torch
from torch import nn

from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules
from ...utils import common_utils
from ...utils.box_coder_utils import IBoxCoder
from ..backbones_3d.spconv_backbone import MultiScale3DFeatures
from ..model_utils.model_nms_utils import NMSConf
from .roi_head_interface import IRoIHead, LossWeights
from .target_assigner import ProposalTargetLayer


@dataclass
class PoolConf:
    name: str
    method: str
    mlps: List[List[int]]
    nsample: List[int]
    radius: List[float]
    ranges: List[List[int]]


class VoxelRCNNHead(IRoIHead):
    def __init__(
        self,
        dp_ratio: float,
        num_class: int,
        cls_channels: List[int],
        reg_channels: List[int],
        shared_channels: List[int],
        pc_range: List[int],
        voxel_size: List[int],
        nms_train_cfg: NMSConf,
        nms_test_cfg: NMSConf,
        pool_cfgs: List[PoolConf],
        pool_grid_size: int,
        box_coder: IBoxCoder,
        proposal_target_layer: ProposalTargetLayer,
        corner_loss_regularization: bool,
        classification_loss_fn: nn.Module,
        reg_loss_fn: nn.Module,
        loss_weights: LossWeights,
    ):
        super().__init__(
            box_coder,
            proposal_target_layer,
            corner_loss_regularization,
            classification_loss_fn,
            reg_loss_fn,
            loss_weights,
        )

        # TODO: in yaml TODO
        self.roi_grid_pool_layers = nn.ModuleList(
            [
                voxel_pool_modules.NeighborVoxelSAModuleMSG(
                    query_ranges=cfg.ranges,
                    nsamples=cfg.nsample,
                    radii=cfg.radius,
                    mlps=cfg.mlps,
                    pool_method=cfg.method,
                )
                for cfg in pool_cfgs
            ]
        )
        self.feat_names = [cfg.name for cfg in pool_cfgs]
        self.nms_train_cfg = nms_train_cfg
        self.nms_test_cfg = nms_test_cfg
        self.pc_range = pc_range
        self.pool_grid_size = pool_grid_size
        self.voxel_size = voxel_size

        c_in = sum([mlp[-1] for cfg in pool_cfgs for mlp in cfg.mlps])
        self.shared_fc_layer = _make_fc_layers(
            in_channels=pool_grid_size**3 * c_in,
            channels=shared_channels,
            dp_ratio=dp_ratio,
            relu_inplace=True,
        )

        self.cls_fc_layers = _make_fc_layers(
            in_channels=shared_channels[-1],
            channels=cls_channels,
            dp_ratio=dp_ratio,
            relu_inplace=False,
        )
        self.cls_pred_layer = nn.Linear(cls_channels[-1], num_class, bias=True)
        self.reg_fc_layers = _make_fc_layers(
            in_channels=cls_channels[-1],
            channels=reg_channels,
            dp_ratio=dp_ratio,
            relu_inplace=False,
        )
        self.reg_pred_layer = nn.Linear(
            reg_channels[-1], self.box_coder.code_size * num_class, bias=True
        )

        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def forward(
        self,
        batch_box_preds: torch.Tensor,
        batch_cls_preds: torch.Tensor,
        gt_boxes: torch.Tensor,
        multiscale_3d_features: MultiScale3DFeatures,
        batch_size: int,
    ):
        """
        :param input_data: input dict
        :return:
        """

        rois, roi_labels, roi_scores, has_class_labels = self.run_nms(
            batch_box_preds=batch_box_preds,
            batch_cls_preds=batch_cls_preds,
            batch_size=batch_size,
            nms_cfg=self.nms_train_cfg if self.training else self.nms_test_cfg,
        )
        if self.training:
            rois, roi_labels = self.assign_targets(
                gt_boxes, rois, roi_labels, roi_scores, batch_size
            )

        # RoI aware pooling (BxN, 6x6x6, C)
        pooled_features = self.roi_grid_pool(multiscale_3d_features, rois, batch_size)
        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     reshape(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features)
        # .transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features)
        # .transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        # TODO consider this function order
        if self.training:
            # TODO move the above code to this scope?
            self.fw_data.rcnn_cls = rcnn_cls
            self.fw_data.rcnn_reg = rcnn_reg
            return rois, roi_labels, has_class_labels
        else:
            batch_box_preds, batch_cls_preds = self.generate_predicted_boxes(
                batch_size, rois, rcnn_cls, rcnn_reg
            )
            return batch_box_preds, batch_cls_preds, has_class_labels

    def roi_grid_pool(
        self, multiscale_3d_features: MultiScale3DFeatures, rois: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        # (BxN, 6x6x6, 3)
        roi_grid_xyz, _ = _get_global_grid_points_of_roi(rois, grid_size=self.pool_grid_size)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div(
            roi_grid_xyz[:, :, 0:1] - self.pc_range[0], self.voxel_size[0], rounding_mode="trunc"
        )
        roi_grid_coords_y = torch.div(
            roi_grid_xyz[:, :, 1:2] - self.pc_range[1], self.voxel_size[1], rounding_mode="trunc"
        )
        roi_grid_coords_z = torch.div(
            roi_grid_xyz[:, :, 2:3] - self.pc_range[2], self.voxel_size[2], rounding_mode="trunc"
        )
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat(
            [roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1
        )

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        # TODO better meaning for loop
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx

        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = torch.full(
            (batch_size,), roi_grid_coords.shape[1], dtype=torch.int32, device=rois.device
        )

        pooled_features_list = []
        for pool_layer, src_name in zip(self.roi_grid_pool_layers, self.feat_names):
            stride_idx = int(src_name[-1])  # TODO better structure
            stride: int = getattr(multiscale_3d_features, f"stride{stride_idx}")
            sp_tensor: spconv.SparseConvTensor = getattr(multiscale_3d_features, src_name)

            # compute voxel center xyz and batch_cnt
            cur_coords = sp_tensor.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.pc_range,
            )
            cur_voxel_xyz_batch_cnt = torch.zeros(
                (batch_size,), dtype=torch.int32, device=cur_voxel_xyz.device
            )
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(sp_tensor)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = torch.div(roi_grid_coords, stride, rounding_mode="trunc")
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=sp_tensor.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor,
            )

            # (BxN, 6x6x6, C)
            pooled_features = pooled_features.view(
                -1, self.pool_grid_size**3, pooled_features.shape[-1]
            )
            pooled_features_list.append(pooled_features)

        return torch.cat(pooled_features_list, dim=-1)


def _get_dense_grid_points(rois, batch_size_rcnn, grid_size):
    faked_features = rois.new_ones((grid_size, grid_size, grid_size))
    dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

    local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
    # (B, 6x6x6, 3)
    roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1)
    return roi_grid_points - local_roi_size.unsqueeze(dim=1) / 2


def _get_global_grid_points_of_roi(rois, grid_size):
    rois = rois.view(-1, rois.shape[-1])
    batch_size_rcnn = rois.shape[0]

    # (B, 6x6x6, 3)
    local_roi_grid_points = _get_dense_grid_points(rois, batch_size_rcnn, grid_size)
    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)
    return global_roi_grid_points, local_roi_grid_points


def _make_fc_layers(in_channels: int, channels: List[int], dp_ratio: float, relu_inplace: bool):
    # TODO py3.10 itertools.pairwise
    in_cs, out_cs = itertools.tee([in_channels, *channels])
    next(out_cs)
    layers = list(
        itertools.chain.from_iterable(
            [
                nn.Linear(in_c, out_c, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=relu_inplace),
                nn.Dropout(dp_ratio) if dp_ratio > 0 else None,
            ]
            for in_c, out_c in zip(in_cs, out_cs)
        )
    )
    return nn.Sequential(*[layer for layer in layers[:-1] if layer is not None])
