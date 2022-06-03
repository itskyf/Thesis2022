from functools import partial

import cumm.tensorview
import numpy as np
from numpy.lib.stride_tricks import as_strided
from omegaconf import ListConfig
from spconv.utils import Point2VoxelCPU3d

from ...utils import box_utils, common_utils


class DataProcessor:
    def __init__(self, cfg: ListConfig, training: bool):
        self.training = training
        self.pc_range = cfg.pc_range
        self.num_point_features = cfg.num_point_features
        self.data_processor_queue = [
            getattr(self, cur_cfg.name)(config=cur_cfg) for cur_cfg in cfg.processor_list
        ]
        self.voxel_generator = None

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get("points", None) is not None:
            mask = common_utils.mask_points_by_range(data_dict["points"], self.pc_range)
            data_dict["points"] = data_dict["points"][mask]

        if (
            data_dict.get("gt_boxes", None) is not None
            and config.remove_outside_boxes
            and self.training
        ):
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict["gt_boxes"],
                self.pc_range,
                min_num_corners=config.get("min_num_corners", 1),
            )
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.shuffle_enabled[self.training]:
            points = data_dict["points"]
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict["points"] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            # Create the VoxelGeneratorWrapper later to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = Point2VoxelCPU3d(
                vsize_xyz=config.voxel_size,
                coors_range_xyz=self.pc_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.max_points_per_voxel,
                max_num_voxels=config.max_number_of_voxels[self.training],
            )

        points = data_dict["points"]
        tv_voxels, tv_coordinates, tv_num_points = self.voxel_generator.point_to_voxel(
            cumm.tensorview.from_numpy(points)
        )

        voxels = tv_voxels.numpy()
        if not data_dict["use_lead_xyz"]:
            # remove xyz in voxels(N, 3)
            voxels = voxels[..., 3:]

        data_dict["voxels"] = voxels
        data_dict["voxel_coords"] = tv_coordinates.numpy()
        data_dict["voxel_num_points"] = tv_num_points.numpy()
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.num_points[self.training]
        if num_points == -1:
            return data_dict

        points = data_dict["points"]
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(
                    near_idxs, num_points - len(far_idxs_choice), replace=False
                )
                choice = (
                    np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
                    if len(far_idxs_choice) > 0
                    else near_idxs_choice
                )
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict["points"] = points[choice]
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.downsample_factor
            return partial(self.downsample_depth_map, config=config)

        data_dict["depth_maps"] = _downscale_local_mean(
            data_dict["depth_maps"],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor),
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


def _downscale_local_mean(image, factors, cval=0):
    """From skimage.transform.downscale_local_mean"""
    if np.isscalar(factors):
        factors = (factors,) * image.ndim
    elif len(factors) != image.ndim:
        raise ValueError(
            "`block_size` must be a scalar or have " "the same length as `image.shape`"
        )

    pad_width = []
    for i in range(len(factors)):
        if factors[i] < 1:
            raise ValueError(
                "Down-sampling factors must be >= 1. Use "
                "`skimage.transform.resize` to up-sample an "
                "image."
            )
        after_width = (
            factors[i] - (image.shape[i] % factors[i]) if image.shape[i] % factors[i] != 0 else 0
        )
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode="constant", constant_values=cval)

    blocked = _view_as_blocks(image, factors)

    return np.mean(blocked, axis=tuple(range(image.ndim, blocked.ndim)))


def _view_as_blocks(arr_in, block_shape):
    """From skimage.util.view_as_blocks"""
    if not isinstance(block_shape, tuple):
        raise TypeError("block needs to be a tuple")

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length " "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out
