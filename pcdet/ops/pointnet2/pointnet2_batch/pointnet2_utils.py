from typing import Optional, Tuple

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
        K = idx.shape[2]
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
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, normalize_dp: bool):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.normalize_dp = normalize_dp

    def forward(
        self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        indices = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()

        grouped_xyz = grouping_operation(xyz_trans, indices)  # (B, 3, K, n_neighbors)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, indices)
            return grouped_xyz, grouped_xyz

        return new_features, None
