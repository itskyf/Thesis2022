from typing import List

import torch
from torch import nn

from ...datasets import PCBatch
from ..backbones_2d import BaseBEVBackbone
from ..backbones_3d import SparseBackboneOut, VoxelBackbone8x
from ..backbones_3d.vfe import MeanVFE
from ..dense_heads.anchor_head_single import AnchorHeadSingle, AnchorHeadSingleOut
from ..roi_heads import IRoIHead
from .detector3d_interface import IDetector3D, NMSConf


class VoxelRCNN(IDetector3D):
    def __init__(
        self,
        vfe: MeanVFE,
        backbone_3d: VoxelBackbone8x,
        map_to_bev: nn.Module,
        backbone_2d: BaseBEVBackbone,
        dense_head: AnchorHeadSingle,
        roi_head: IRoIHead,
        output_raw_score: bool,
        nms_cfgs: NMSConf,
        recall_thresholds: List[float],
    ):
        super().__init__(output_raw_score, nms_cfgs, recall_thresholds)
        self.vfe = vfe
        self.backbone_3d = backbone_3d
        self.map_to_bev = map_to_bev
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head
        self.roi_head = roi_head

    def forward(self, pc_batch: PCBatch):
        voxel_features: torch.Tensor = self.vfe(pc_batch.voxels, pc_batch.voxel_num_points)
        bb3d_out: SparseBackboneOut = self.backbone_3d(
            voxel_features, pc_batch.voxel_coords, pc_batch.batch_size
        )
        spatial_features: torch.Tensor = self.map_to_bev(bb3d_out.sparse_out)
        spatial_features_2d: torch.Tensor = self.backbone_2d(spatial_features)
        dense_out: AnchorHeadSingleOut = self.dense_head(
            spatial_features_2d, pc_batch.gt_boxes, pc_batch.batch_size
        )
        box_preds, cls_preds, has_class_labels = self.roi_head(
            batch_box_preds=dense_out.batch_box_preds,
            batch_cls_preds=dense_out.batch_cls_preds,
            gt_boxes=pc_batch.gt_boxes,
            multiscale_3d_features=bb3d_out.multiscale_3d_features,
            batch_size=pc_batch.batch_size,
        )

        if self.training:
            cls_loss, dir_loss, loc_loss = self.dense_head.get_loss()
            rcnn_cls_loss, rcnn_corner_loss, rcnn_reg_loss = self.roi_head.get_loss()
            return {
                "cls_loss": cls_loss,
                "dir_loss": dir_loss,
                "loc_loss": loc_loss,
                "rcnn_cls_loss": rcnn_cls_loss,
                "rcnn_corner_loss": rcnn_corner_loss,
                "rcnn_reg_loss": rcnn_reg_loss,
            }
        else:
            preds, recall_dict = self.post_processing(
                batch_box_preds=dense_out.batch_box_preds,
                batch_cls_preds=dense_out.batch_cls_preds,
                gt_boxes=pc_batch.gt_boxes,
                rois=box_preds,
                roi_labels=cls_preds,
                batch_size=pc_batch.batch_size,
                has_class_labels=has_class_labels,
            )
            return preds, recall_dict
