from typing import Optional

import torch
from torch import nn

from ..pfe.pointnet2_sa import PointNetSAMSGSampling
from ..pfe.pointnet2_sa_ln import PointNetSAMSGSamplingLN


class IASSDEncoder(nn.Module):
    """Backbone for IA-SSD"""

    def __init__(
        self,
        n_class: int,
        in_channels: int,
        confidence_cs: list[Optional[int]],
        n_points: list[int],
        n_samples: list[list[int]],
        radii_list: list[list[float]],
        mlps_channels_list: list[list[list[int]]],
        out_channels_list: list[int],
    ):
        assert (
            len(confidence_cs)
            == len(n_points)
            == len(radii_list)
            == len(n_samples)
            == len(mlps_channels_list)
            == len(out_channels_list)
        )
        super().__init__()
        in_cs_list = (in_channels, *out_channels_list[:-1])
        sa_layers = [
            PointNetSAMSGSampling(n_class, in_cs, conf_cs, n_pt, n_sample, radii, mlps_cs, out_cs)
            for in_cs, conf_cs, n_pt, n_sample, radii, mlps_cs, out_cs in zip(
                in_cs_list,
                confidence_cs,
                n_points,
                n_samples,
                radii_list,
                mlps_channels_list,
                out_channels_list,
            )
        ]
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, batch_size: int, pcd_pts: torch.Tensor):
        """
        Args:
            points:
        Returns:
            batch_dict:
        """
        points, feats = _split_xyz(pcd_pts)
        points = points.view(batch_size, -1, 3)
        feats = feats.view(batch_size, -1, feats.size(-1)).transpose(1, 2).contiguous()

        pts_list = []
        cls_preds_list = []
        for sa_layer in self.sa_layers:
            points, feats, cls_preds = sa_layer(points, feats)
            pts_list.append(points)
            cls_preds_list.append(cls_preds)
        return pts_list, feats, cls_preds_list


class IASSDEncLN(nn.Module):
    """Backbone for IA-SSD"""

    def __init__(
        self,
        n_class: int,
        in_channels: int,
        confidence_cs: list[Optional[int]],
        n_points: list[int],
        n_samples: list[list[int]],
        radii_list: list[list[float]],
        mlps_channels_list: list[list[list[int]]],
        out_channels_list: list[int],
    ):
        assert (
            len(confidence_cs)
            == len(n_points)
            == len(radii_list)
            == len(n_samples)
            == len(mlps_channels_list)
            == len(out_channels_list)
        )
        super().__init__()
        in_cs_list = (in_channels, *out_channels_list[:-1])
        sa_layers = [
            PointNetSAMSGSamplingLN(n_class, in_cs, conf_cs, n_pt, n_sample, radii, mlps_cs, out_cs)
            for in_cs, conf_cs, n_pt, n_sample, radii, mlps_cs, out_cs in zip(
                in_cs_list,
                confidence_cs,
                n_points,
                n_samples,
                radii_list,
                mlps_channels_list,
                out_channels_list,
            )
        ]
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, batch_size: int, pcd_pts: torch.Tensor):
        """
        Args:
            points:
        Returns:
            batch_dict:
        """
        points, feats = _split_xyz(pcd_pts)
        points = points.view(batch_size, -1, 3)
        feats = feats.view(batch_size, -1, feats.size(-1)).transpose(1, 2).contiguous()

        pts_list = []
        cls_preds_list = []
        for sa_layer in self.sa_layers:
            points, feats, cls_preds = sa_layer(points, feats)
            pts_list.append(points)
            cls_preds_list.append(cls_preds)
        return pts_list, feats, cls_preds_list


def _split_xyz(pcd: torch.Tensor):
    points = pcd[:, 1:4].contiguous()
    feats = pcd[:, 4:].contiguous()
    return points, feats
