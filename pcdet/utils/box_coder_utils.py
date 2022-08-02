import numpy as np
import torch
from torch.nn import functional


class PointResidualBinOriCoder:
    def __init__(self, bin_size: int):
        super().__init__()
        self.bin_size = bin_size
        # self.bin_size = 12
        self.code_size = 6 + 2 * self.bin_size
        self.bin_inter = 2 * np.pi / self.bin_size

    def encode_torch(
        self,
        gt_boxes: torch.Tensor,
        points: torch.Tensor,
        gt_classes: torch.Tensor,
        mean_size: torch.Tensor,
    ):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[..., 3:6] = torch.clamp_min(gt_boxes[..., 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        point_anchor_size = mean_size[gt_classes - 1]
        dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
        diagonal = torch.sqrt(dxa**2 + dya**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)

        rg = torch.clamp(rg, max=np.pi - 1e-5, min=-np.pi + 1e-5)
        bin_id = torch.floor((rg + np.pi) / self.bin_inter)
        bin_res = (rg + np.pi) - (bin_id * self.bin_inter + self.bin_inter / 2)
        bin_res /= self.bin_inter / 2
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, bin_id, bin_res, *cgs], dim=-1)

    def decode_torch(
        self,
        box_encodings: torch.Tensor,
        points: torch.Tensor,
        pred_classes: torch.Tensor,
        mean_size: torch.Tensor,
    ):
        """
        Args:
            box_encodings: (B, N 8 + C) [x, y, z, dx, dy, dz, bin_id, bin_res , ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
        """
        xt, yt, zt, dxt, dyt, dzt = torch.split(box_encodings[..., :6], 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        point_anchor_size = mean_size[pred_classes - 1]
        dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
        diagonal = torch.sqrt(dxa**2 + dya**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        bin_id = box_encodings[..., 6 : 6 + self.bin_size]
        bin_res = box_encodings[..., 6 + self.bin_size :]
        _, bin_id = torch.max(bin_id, dim=-1)
        bin_id_one_hot = functional.one_hot(bin_id.long(), self.bin_size)
        bin_res = torch.sum(bin_res * bin_id_one_hot.float(), dim=-1)

        rg = bin_id.float() * self.bin_inter - np.pi + self.bin_inter / 2
        rg = rg + bin_res * (self.bin_inter / 2)
        rg = rg.unsqueeze(-1)
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)
