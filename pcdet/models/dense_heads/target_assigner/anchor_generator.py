from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class AnchorConf:
    class_name: str
    anchor_sizes: List[List[int]]
    anchor_rotations: List[int]
    anchor_bottom_heights: List[int]
    align_center: bool
    feature_map_stride: int
    matched_threshold: float
    unmatched_threshold: float


class AnchorGenerator:
    def __init__(self, anchor_range: List[int], cfgs: List[AnchorConf]):
        super().__init__()
        self.anchor_range = anchor_range
        # TODO better configurations
        self.align_center = [cfg.align_center for cfg in cfgs]
        self.anchor_heights = [cfg.anchor_bottom_heights for cfg in cfgs]
        self.anchor_rotations = [cfg.anchor_rotations for cfg in cfgs]
        self.anchor_sizes = [cfg.anchor_sizes for cfg in cfgs]

        assert len(self.anchor_sizes) == len(self.anchor_rotations)
        assert len(self.anchor_sizes) == len(self.anchor_heights)

    def generate_anchors(self, grid_sizes: npt.NDArray[np.int32]) -> Tuple[List, List[int]]:
        assert len(grid_sizes) == len(self.anchor_sizes)
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
            grid_sizes,
            self.anchor_sizes,
            self.anchor_rotations,
            self.anchor_heights,
            self.align_center,
        ):
            num_anchors_per_location.append(
                len(anchor_rotation) * len(anchor_size) * len(anchor_height)
            )
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            cuda = torch.device("cuda")
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset,
                self.anchor_range[3] + 1e-5,
                step=x_stride,
                dtype=torch.float32,
                device=cuda,
            )
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset,
                self.anchor_range[4] + 1e-5,
                step=y_stride,
                dtype=torch.float32,
                device=cuda,
            )
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = len(anchor_size), len(anchor_rotation)
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            # [x_grid, y_grid, z_grid]
            x_shifts, y_shifts, z_shifts = torch.meshgrid(
                [x_shifts, y_shifts, z_shifts], indexing="ij"
            )
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)

            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1)
            anchor_rotation = anchor_rotation.repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            # [x, y, z, num_size, num_rot, 7]
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            # anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location
