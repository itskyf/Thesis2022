from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .box_utils import boxes_to_corners_3d


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(pred: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        return torch.clamp(pred, min=0) - pred * target + torch.log1p(torch.exp(-torch.abs(pred)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(pred)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)

        loss = focal_weight * bce_loss
        if weights.dim() == 2 or (weights.dim() == 1 and target.dim() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.dim() == loss.dim()
        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = pred - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            pred: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), pred, target)  # ignore nan targets
        # code-wise weighting
        diff = pred - target
        loss = self.smooth_l1_loss(diff, self.beta)
        # anchor-wise weighting
        if weights is not None:
            assert weights.size(0) == loss.size(0) and weights.size(1) == loss.size(1)
            loss = loss * weights.unsqueeze(-1)

        return loss

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            abs_diff = torch.abs(diff)
            loss = torch.where(abs_diff < beta, 0.5 * abs_diff**2 / beta, abs_diff - 0.5 * beta)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super().__init__()
        if code_weights is not None:
            self.register_buffer(
                "code_weights", torch.tensor(code_weights, dtype=torch.float32).view(1, 1, -1)
            )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            pred: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), pred, target)  # ignore nan targets

        try:
            # code-wise weighting
            diff = self.get_buffer("code_weights") * (pred - target)
        except AttributeError:
            diff = pred - target

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.size(0) == loss.size(0) and weights.size(1) == loss.size(1)
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        pred = pred.transpose(1, 2)
        target = target.argmax(dim=-1)
        return functional.cross_entropy(pred, target, reduction="none") * weights


class WeightedClassificationLoss(nn.Module):
    @staticmethod
    def sigmoid_cross_entropy_with_logits(pred: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        return torch.clamp(pred, min=0) - pred * target + torch.log1p(torch.exp(-torch.abs(pred)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights=None, reduction="none"):
        """
        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)

        if weights is not None:
            if weights.dim() == 2 or (weights.dim() == 1 and target.dim() == 2):
                weights = weights.unsqueeze(-1)

            assert weights.dim() == bce_loss.dim()

            loss = weights * bce_loss
        else:
            loss = bce_loss

        if reduction == "sum":
            return loss.sum(dim=-1)

        if reduction == "mean":
            return loss.mean(dim=-1)

        return loss


class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            pred: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.
        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        loss = torch.mean(
            functional.binary_cross_entropy_with_logits(pred, target, reduction="none"), dim=-1
        )
        return loss * weights


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.size(0) == gt_bbox3d.size(0)

    pred_box_corners = boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(
        torch.norm(pred_box_corners - gt_box_corners, dim=2),
        torch.norm(pred_box_corners - gt_box_corners_flip, dim=2),
    )
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    b, n = gt_boxes2d.shape[:2]
    for b in range(b):
        for n in range(n):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    return (loss - neg_loss) if num_pos == 0 else (loss - (pos_loss + neg_loss) / num_pos)


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super().__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    return loss / torch.clamp_min(num, min=1.0)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        pred = output if ind is None else _transpose_and_gather_feat(output, ind)
        return _reg_loss(pred, target, mask)
