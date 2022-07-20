from typing import Optional

import torch
from torch import nn

from ....ops.pointnet2 import _C, grouping_ops, masked_gather
from ...modules import LayerNorm1d, LayerNorm2d


class QueryAndGroup(nn.Module):
    def __init__(self, n_sample: int, radius: float, normalize_dp: bool):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.n_sample = n_sample
        self.radius = radius
        self.normalize_dp = normalize_dp

    def forward(
        self, points: torch.Tensor, centroids: torch.Tensor, features: Optional[torch.Tensor] = None
    ):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, K, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            grouped_pts: (B, 3, K, n_sample)
            new_features: (B, C, K, n_sample)
        """
        indices = _C.ball_query(points, centroids, self.n_sample, self.radius)

        grouped_pts = grouping_ops(points.transpose(1, 2), indices)  # (B, 3, K, n_sample)
        grouped_pts -= centroids.transpose(1, 2).unsqueeze(-1)
        if self.normalize_dp:
            grouped_pts /= self.radius

        if features is not None:
            new_feats = grouping_ops(features, indices)
            return grouped_pts, new_feats

        return grouped_pts, None

    def extra_repr(self):
        return f"radius={self.radius}, n_sample={self.n_sample}, normalize_dp={self.normalize_dp}"


class PointNetSAMSGLN(nn.Module):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping"""

    def __init__(
        self,
        in_channels: int,
        n_sample: list[int],
        radii: list[float],
        mlps_cs: list[list[int]],
        out_channels: int,
        normalize_dp: bool = True,
        pool_method: str = "max",
        use_xyz: bool = True,
    ):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        assert len(radii) == len(n_sample) == len(mlps_cs)
        assert len(mlps_cs[0]) == 3
        assert pool_method in ("max")
        super().__init__()
        self.groupers = nn.ModuleList(
            [
                QueryAndGroup(n_sample, radius, normalize_dp)
                for n_sample, radius in zip(n_sample, radii)
            ]
        )

        self.use_xyz = use_xyz
        if self.use_xyz:
            in_channels += 3
        mlps = [
            nn.Sequential(
                nn.Conv2d(in_channels, mlp_cs[0], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[0], data_format="channels_first"),
                nn.Conv2d(mlp_cs[0], mlp_cs[1], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[1], data_format="channels_first"),
                nn.GELU(),
                nn.Conv2d(mlp_cs[1], mlp_cs[2], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[2], data_format="channels_first"),
            )
            for mlp_cs in mlps_cs
        ]
        self.mlps = nn.ModuleList(mlps)
        self.pool_layer = lambda x: torch.max(x, dim=-1, keepdim=False).values

        mlps_out = sum(mlp_cs[-1] for mlp_cs in mlps_cs)
        self.aggregation_layer = nn.Sequential(
            nn.Conv1d(mlps_out, out_channels, kernel_size=1, bias=False),
            LayerNorm1d(out_channels, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor, centroids: torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        feats_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_pts, new_feats = grouper(points, centroids, features)
            # (B, C, npoint, nsample)
            if self.use_xyz:
                new_feats = torch.cat((grouped_pts, new_feats), dim=1)
            new_feats = mlp(new_feats)  # (B, mlp[-1], npoint, nsample)
            new_feats = self.pool_layer(new_feats)
            feats_list.append(new_feats)
        new_feats = torch.cat(feats_list, dim=1)
        return self.aggregation_layer(new_feats)


class PointNetSAMSGSamplingLN(nn.Module):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping"""

    def __init__(
        self,
        n_class: int,
        in_channels: int,
        confidence_cs: Optional[int],
        n_point: int,
        n_sample: list[int],
        radii: list[float],
        mlps_cs: list[list[int]],
        out_channels: int,
        normalize_dp: bool = True,
        pool_method: str = "max",
        use_xyz: bool = True,
    ):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        assert len(radii) == len(n_sample) == len(mlps_cs)
        assert len(mlps_cs[0]) == 3
        assert pool_method in ("max")
        super().__init__()
        if confidence_cs is not None:
            self.confidence_layer = nn.Sequential(
                nn.Conv1d(in_channels, confidence_cs, kernel_size=1, bias=False),
                LayerNorm1d(confidence_cs, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(confidence_cs, n_class, kernel_size=1, bias=True),
            )
        self.n_point = n_point
        self.groupers = nn.ModuleList(
            [
                QueryAndGroup(n_sample, radius, normalize_dp)
                for n_sample, radius in zip(n_sample, radii)
            ]
        )

        self.use_xyz = use_xyz
        if self.use_xyz:
            in_channels += 3
        mlps = [
            nn.Sequential(
                nn.Conv2d(in_channels, mlp_cs[0], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[0], data_format="channels_first"),
                nn.Conv2d(mlp_cs[0], mlp_cs[1], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[1], data_format="channels_first"),
                nn.GELU(),
                nn.Conv2d(mlp_cs[1], mlp_cs[2], kernel_size=1, bias=False),
                LayerNorm2d(mlp_cs[2], data_format="channels_first"),
            )
            for mlp_cs in mlps_cs
        ]
        self.mlps = nn.ModuleList(mlps)
        self.pool_layer = lambda x: torch.max(x, dim=-1, keepdim=False).values

        mlps_out = sum(mlp_cs[-1] for mlp_cs in mlps_cs)
        self.aggregation_layer = nn.Sequential(
            nn.Conv1d(mlps_out, out_channels, kernel_size=1, bias=False),
            LayerNorm1d(out_channels, data_format="channels_first"),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        assert points.size(1) >= self.n_point
        try:
            cls_preds = self.confidence_layer(features)
            cls_features_max = cls_preds.max(dim=1).values
            score_pred = torch.sigmoid(cls_features_max)
            sample_indices = torch.topk(score_pred, self.n_point, dim=-1).indices
        except AttributeError:
            cls_preds = None
            sample_indices = _C.farthest_point_sampling(points, self.n_point)
            sample_indices = sample_indices.long()
        new_points = masked_gather(points, sample_indices)

        feats_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_pts, new_feats = grouper(points, new_points, features)
            # (B, C, npoint, nsample)
            if self.use_xyz:
                new_feats = torch.cat((grouped_pts, new_feats), dim=1)
            new_feats = mlp(new_feats)  # [B, mlp[-1], npoint, nsample]
            new_feats = self.pool_layer(new_feats)  # [B, mlp[-1], npoint]
            feats_list.append(new_feats)
        new_feats = torch.cat(feats_list, dim=1)
        new_feats = self.aggregation_layer(new_feats)
        return new_points, new_feats, cls_preds
