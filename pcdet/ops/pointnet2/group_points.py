from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from . import pointnet2_batch_cuda as pointnet2


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.
    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding
    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.size(2)
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor):
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param indices: (B, K, n_neighbors) indicies of features to group with
        :return:
            output: (B, C, K, n_neighbors) tensor
        """
        total_pts = features.size(2)
        ctx.for_backwards = (indices, total_pts)
        return group_points(features, indices)

    @staticmethod
    def backward(ctx, grad_grouped: torch.Tensor):
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        indices, total_pts = ctx.for_backwards
        grad_features: torch.Tensor = pointnet2.group_points_backward(
            grad_grouped, indices, total_pts
        )
        grad_features.requires_grad_(True)
        return grad_features, None


grouping_ops = GroupingOperation.apply
