import torch
from torch import nn

from ....ops.pointnet2 import masked_gather


class VoteLayer(nn.Module):
    """Light voting module with limitation"""

    def __init__(
        self,
        n_class: int,
        in_channels: int,
        n_point: int,
        mid_channels: int,
        max_offset_limit: list[int],
    ):
        super().__init__()
        self.confidence_layer = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, n_class, kernel_size=1, bias=True),
        )
        self.n_point = n_point
        self.ctr_reg = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, 3, kernel_size=1),
        )
        self.register_buffer(
            "max_offset_limit", torch.tensor(max_offset_limit).view(1, 1, 3), persistent=False
        )

    def forward(self, points: torch.Tensor, feats: torch.Tensor):
        """feats [B, C, N]"""
        cls_preds = self.confidence_layer(feats)

        cls_features_max = cls_preds.max(dim=1).values
        score_pred = torch.sigmoid(cls_features_max)
        sample_indices = torch.topk(score_pred, self.n_point, dim=-1).indices

        ctr_origins = masked_gather(points, sample_indices)
        feats = masked_gather(feats.transpose(1, 2), sample_indices)

        ctr_offsets = self.ctr_reg(feats.transpose(1, 2))
        ctr_offsets = ctr_offsets.transpose(1, 2)  # [B, 256, 3]

        max_offset_limit = self.get_buffer("max_offset_limit")
        # [B, 256, 3]
        max_offset_limit = max_offset_limit.expand((ctr_origins.size(0), ctr_origins.size(1), -1))

        limited_ctr_offsets = torch.where(
            ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets
        )
        min_offset_limit = torch.neg(max_offset_limit)
        limited_ctr_offsets = torch.where(
            limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets
        )
        ctr_preds = ctr_origins + limited_ctr_offsets
        return cls_preds, ctr_preds, ctr_origins, ctr_offsets
