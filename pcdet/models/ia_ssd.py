import torch
from torch import nn
from torch.nn import functional

from ..utils import common_utils, loss_utils
from .backbones_3d.ia_ssd import IASSDEncoder, VoteLayer
from .backbones_3d.pfe import PointNetSAMSG
from .dense_heads.ia_ssd_head import IASSDHead, TrainTargets
from .post_process import post_processing


class IASSDNet(nn.Module):
    def __init__(self, model_cfg, n_class: int):
        super().__init__()
        self.encoder = IASSDEncoder(n_class, **model_cfg.enc_cfg)
        self.vote_layer = VoteLayer(n_class, **model_cfg.vote_cfg)
        self.ctr_agg_layer = PointNetSAMSG(**model_cfg.ctr_agg_cfg)
        self.point_head = IASSDHead(n_class, **model_cfg.head_cfg)

        self.n_class = n_class
        self.nms_cfg = model_cfg.nms_cfg
        self.post_cfg = model_cfg.post_process_cfg
        self.loss_weights = model_cfg.loss_weights

    def forward(self, batch_dict):
        batch_size = batch_dict["batch_size"]
        points = batch_dict["points"]

        pts_list, feats, cls_preds_list = self.encoder(batch_size, points)
        cls_preds, ctr_preds, ctr_origins, ctr_offsets = self.vote_layer(pts_list[-1], feats)
        ctr_feats = self.ctr_agg_layer(pts_list[-1], feats, ctr_preds)

        gt_boxes = batch_dict["gt_boxes"]
        # TODO if traning, only assign targets for points before cls_preds
        ctr_cls_preds, ctr_box_preds, pt_box_preds, targets = self.point_head(
            ctr_feats, ctr_preds, ctr_origins, pts_list, gt_boxes
        )
        if targets is None:
            return post_processing(
                ctr_cls_preds, pt_box_preds, gt_boxes, self.n_class, self.nms_cfg, self.post_cfg
            )

        targets: TrainTargets  # Training
        ctr_t = targets.center
        assert ctr_t.pt_box_labels is not None
        pt_xyzwhl_loss, ori_cls_loss, ori_reg_loss = _center_box_binori_loss(
            ctr_box_preds,
            ctr_t.pt_box_labels,
            ctr_t.pt_cls_labels,
            self.point_head.box_coder.bin_size,
        )
        ori_cls_loss *= self.loss_weights.direction
        center_reg_loss = pt_xyzwhl_loss + ori_cls_loss + ori_reg_loss
        center_reg_loss *= self.loss_weights.center_reg

        ctr_cls_loss = _classification_loss(
            ctr_cls_preds, ctr_t.pt_cls_labels, ctr_preds, ctr_t.gt_box_of_fg_pts
        )
        ctr_cls_loss *= self.loss_weights.classification

        corner_loss = _corner_loss(pt_box_preds, ctr_t.gt_box_of_fg_pts, ctr_t.pt_cls_labels)
        corner_loss *= self.loss_weights.corner
        # Semantic loss in SA layers
        sa_ins_preds = []
        sa_ins_labels = []
        sa_ins_fg_gt_boxes = []
        sa_pts_list = []
        for sa_pts, target, preds in zip(pts_list, targets.sa_ins, cls_preds_list[1:]):
            if preds is not None:
                sa_ins_preds.append(preds)
                sa_pts_list.append(sa_pts)
                sa_ins_labels.append(target.pt_cls_labels)
                sa_ins_fg_gt_boxes.append(target.gt_box_of_fg_pts)
        sa_ins_preds.append(cls_preds)
        sa_pts_list.append(pts_list[-1])
        sa_ins_labels.append(targets.sa_ins[-1].pt_cls_labels)
        sa_ins_fg_gt_boxes.append(targets.sa_ins[-1].gt_box_of_fg_pts)

        sa_cls_losses = [
            _classification_loss(sa_preds.transpose(1, 2), sa_labels, sa_pts, sa_fg_gt_boxes)
            for sa_preds, sa_labels, sa_pts, sa_fg_gt_boxes in zip(
                sa_ins_preds,
                sa_ins_labels,
                sa_pts_list,
                sa_ins_fg_gt_boxes,
            )
        ]
        sa_cls_losses = [
            loss * weight for loss, weight in zip(sa_cls_losses, self.loss_weights.ins_aware)
        ]
        sa_cls_loss = sum(sa_cls_losses) / len(sa_cls_losses)

        ctr_org_t = targets.ctr_origin
        vote_loss = _contextual_vote_loss(
            ctr_origins, ctr_offsets, ctr_org_t.gt_box_of_fg_pts, ctr_org_t.pt_cls_labels
        )
        vote_loss *= self.loss_weights.voting

        return {
            "center_reg": center_reg_loss,
            "center_classification": ctr_cls_loss,
            "corner": corner_loss,
            "voting": vote_loss,
            "sa_classification": sa_cls_loss,
        }


def _center_box_binori_loss(
    ctr_box_preds: torch.Tensor,
    fg_ctr_box_labels: torch.Tensor,
    ctr_cls_labels: torch.Tensor,
    bin_size: int,
):
    """
    Args:
        ctr_box_preds: FloatTensor [B, N, 30]
        fg_ctr_box_label: FloatTensor [B, N, 8]
        ctr_cls_labels: LongTensor [B, N] category labels of each points
    """
    pos_mask = ctr_cls_labels > 0
    reg_weights = pos_mask.float()
    reg_weights /= torch.clamp(pos_mask.sum(), min=1.0)

    pred_box_xyzwhl = ctr_box_preds[..., :6]
    label_box_xyzwhl = fg_ctr_box_labels[..., :6]
    pt_box_src_loss = functional.smooth_l1_loss(
        pred_box_xyzwhl, label_box_xyzwhl, reduction="none", beta=1 / 9
    )
    pt_xyzwhl_loss = torch.sum(pt_box_src_loss * reg_weights.unsqueeze(-1))

    pred_ori_bin_id = ctr_box_preds[..., 6 : 6 + bin_size].transpose(1, 2)
    label_ori_bin_id = fg_ctr_box_labels[..., 6].long()
    ori_cls_loss = functional.cross_entropy(pred_ori_bin_id, label_ori_bin_id, reduction="none")
    ori_cls_loss = torch.sum(ori_cls_loss * reg_weights)

    pred_ori_bin_res = ctr_box_preds[..., 6 + bin_size :]
    label_ori_bin_res = fg_ctr_box_labels[..., 7]
    label_id_one_hot = functional.one_hot(label_ori_bin_id, num_classes=bin_size)
    pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
    ori_reg_loss = functional.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
    ori_reg_loss = torch.sum(ori_reg_loss * reg_weights)

    return pt_xyzwhl_loss, ori_cls_loss, ori_reg_loss


def _classification_loss(
    pt_cls_preds: torch.Tensor,
    pt_cls_labels: torch.Tensor,
    ctr_preds: torch.Tensor,
    ctr_gt_boxes: torch.Tensor,
):
    """
    Args:
        pt_cls_preds: FloatTensor [B, N, num_classes]
        pt_cls_labels: LongTensor [B, N] category labels of each points
        ctr_preds: FloatTensor [B, N, 3]
        ctr_gt_boxes: FloatTensor [NFG, 8]
    """
    pos_mask = pt_cls_labels > 0
    neg_mask = pt_cls_labels == 0
    num_classes = pt_cls_preds.size(-1)

    cls_weights = neg_mask + pos_mask.float()
    cls_weights /= torch.clamp(pos_mask.sum(), min=1.0)
    cls_weights = cls_weights.unsqueeze(-1).expand(-1, -1, num_classes)

    fg_cls_labels = pt_cls_labels.clone()
    fg_cls_labels[fg_cls_labels < 0] = 0
    # Centerness regularization
    centerness_mask = _generate_centerness_mask(pos_mask, ctr_preds, ctr_gt_boxes)
    one_hot_targets = functional.one_hot(fg_cls_labels, num_classes + 1)
    # Remove first column since 0 is label of background
    one_hot_targets = one_hot_targets[..., 1:].float()
    one_hot_targets *= centerness_mask.unsqueeze(-1)

    pts_cls_loss = functional.binary_cross_entropy_with_logits(
        pt_cls_preds, one_hot_targets, cls_weights, reduction="none"
    )
    return pts_cls_loss.mean(dim=-1).sum()


def _contextual_vote_loss(
    ctr_origins: torch.Tensor,
    ctr_offsets: torch.Tensor,
    ctr_org_fg_pt_gt_boxes: torch.Tensor,
    ctr_org_cls_labels: torch.Tensor,
):
    """
    Args:
        ctr_origins: FloatTensor [B, N, 3]
        ctr_offsets: FloatTensor [B, N, 3]
        ctr_org_fg_pt_gt_boxes: FloatTensor [NFG, 7]
        ctr_org_cls_labels: LongTensor [B, N]
    """
    pos_mask = ctr_org_cls_labels > 0
    center_org_box_losses = []
    for label in ctr_org_cls_labels.unique():
        if label <= 0:
            continue
        label_pos_mask = ctr_org_cls_labels == label
        mask = pos_mask & label_pos_mask

        center_box_labels = ctr_org_fg_pt_gt_boxes[:, 0:3][mask[pos_mask]]
        centers_pred = ctr_origins + ctr_offsets
        centers_pred = centers_pred[label_pos_mask]

        label_ctr_org_loss_box = functional.smooth_l1_loss(centers_pred, center_box_labels)
        center_org_box_losses.append(label_ctr_org_loss_box.unsqueeze(0))
    center_org_box_losses = torch.cat(center_org_box_losses)
    return center_org_box_losses.mean()


def _corner_loss(
    pt_box_preds: torch.Tensor, fg_ctr_gt_boxes: torch.Tensor, ctr_cls_labels: torch.Tensor
):
    """
    Args:
        pt_box_preds: FloatTensor [B, N, 7]
        fg_ctr_gt_boxes: FloatTensor [NFG, 7]
        ctr_cls_labels: LongTensor [B, N]
    """
    pos_mask = ctr_cls_labels > 0
    pt_box_preds = pt_box_preds[pos_mask]
    corner_loss = loss_utils.get_corner_loss_lidar(pt_box_preds, fg_ctr_gt_boxes)
    return corner_loss.mean()


def _generate_centerness_mask(pos_mask: torch.Tensor, points: torch.Tensor, gt_boxes: torch.Tensor):
    """
    Args:
        pos_mask: BoolTensor [B, N]
        points: FloatTensor [B, N, 3]
        gt_boxes: FloatTensor [NFG, 7]
    Returns:
        centerness_mask: FloatTensor [B, N]
    """
    # TODO medium using MMDetection3D’s centerness_loss style
    # NFG, 3
    points = points[pos_mask].detach()

    offset_xyz = points[:, :3] - gt_boxes[:, :3]
    canonical_xyz = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6])

    template = gt_boxes.new_tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    margin = gt_boxes[:, None, 3:6].expand(-1, 2, -1) * template.unsqueeze(0)
    distance = margin - canonical_xyz.expand(-1, 2, -1)

    dist_0 = distance[:, 0, :]
    dist_1 = torch.neg(distance[:, 1, :])
    distance_min = torch.where(dist_0 < dist_1, dist_0, dist_1)
    distance_max = torch.where(dist_0 > dist_1, dist_0, dist_1)

    centerness = distance_min / distance_max
    centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
    centerness = torch.clamp(centerness, min=1e-6)
    centerness = centerness.pow(1 / 3)

    centerness_mask = torch.zeros_like(pos_mask, dtype=torch.float32)
    centerness_mask[pos_mask] = centerness
    return centerness_mask
