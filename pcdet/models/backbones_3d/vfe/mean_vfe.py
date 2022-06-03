import torch
from torch import nn


class MeanVFE(nn.Module):
    def forward(self, voxels: torch.Tensor, voxel_num_points: torch.Tensor):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0)
        points_mean = voxels.sum(dim=1) / normalizer.type_as(voxels)
        return points_mean.contiguous()
