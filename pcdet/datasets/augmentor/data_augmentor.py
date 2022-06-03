from functools import partial

import numpy as np
from omegaconf import ListConfig, OmegaConf

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor:
    def __init__(self, cfgs: ListConfig):
        # TODO: refactor each method as a class
        self.data_augmentor_queue = []
        for cur_cfg in cfgs:
            cur_augmentor = getattr(self, cur_cfg.name)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        return database_sampler.DataBaseSampler(config)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for cur_axis in config.along_axis_list:
            assert cur_axis in ["x", "y"]
            gt_boxes, points = getattr(augmentor_utils, f"random_flip_along_{cur_axis}")(
                gt_boxes, points
            )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = OmegaConf.to_container(config.world_rot_angle)
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict["gt_boxes"], data_dict["points"], rot_range=rot_range
        )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict["gt_boxes"], data_dict["points"], config.world_scale_range
        )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict["gt_boxes"]
        calib = data_dict["calib"]
        for cur_axis in config.along_axis_list:
            assert cur_axis in ["horizontal"]
            images, depth_maps, gt_boxes = getattr(
                augmentor_utils, f"random_image_flip_{cur_axis}"
            )(images, depth_maps, gt_boxes, calib)

        data_dict["images"] = images
        data_dict["depth_maps"] = depth_maps
        data_dict["gt_boxes"] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config.noise_translate_std
        if noise_translate_std == 0:
            return data_dict
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for cur_axis in config.along_axis_list:
            assert cur_axis in ["x", "y", "z"]
            gt_boxes, points = getattr(augmentor_utils, f"random_translation_along_{cur_axis}")(
                gt_boxes, points, noise_translate_std
            )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config.local_translation_range
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for cur_axis in config.along_axis_list:
            assert cur_axis in ["x", "y", "z"]
            gt_boxes, points = getattr(
                augmentor_utils, f"random_local_translation_along_{cur_axis}"
            )(gt_boxes, points, offset_range)

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config.local_rot_angle
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict["gt_boxes"], data_dict["points"], rot_range=rot_range
        )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict["gt_boxes"], data_dict["points"], config.local_scale_range
        )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config.intensity_range
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for direction in config.direction:
            assert direction in ["top", "bottom", "left", "right"]
            gt_boxes, points = getattr(augmentor_utils, f"global_frustum_dropout_{direction}")(
                gt_boxes, points, intensity_range
            )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config.intensity_range
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for direction in config.direction:
            assert direction in ["top", "bottom", "left", "right"]
            gt_boxes, points = getattr(augmentor_utils, f"local_frustum_dropout_{direction}")(
                gt_boxes, points, intensity_range
            )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(
            gt_boxes, points, config.drop_prob
        )
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(
            gt_boxes, points, config.sparsify_prob, config.sparsify_max_num, pyramids
        )
        gt_boxes, points = augmentor_utils.local_pyramid_swap(
            gt_boxes, points, config.swap_prob, config.swap_max_num, pyramids
        )
        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict["gt_boxes"][:, 6] = common_utils.limit_period(
            data_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi
        )
        if "calib" in data_dict:
            data_dict.pop("calib")
        if "road_plane" in data_dict:
            data_dict.pop("road_plane")
        if "gt_boxes_mask" in data_dict:
            gt_boxes_mask = data_dict["gt_boxes_mask"]
            data_dict["gt_boxes"] = data_dict["gt_boxes"][gt_boxes_mask]
            data_dict["gt_names"] = data_dict["gt_names"][gt_boxes_mask]
            if "gt_boxes2d" in data_dict:
                data_dict["gt_boxes2d"] = data_dict["gt_boxes2d"][gt_boxes_mask]
            data_dict.pop("gt_boxes_mask")
        return data_dict
