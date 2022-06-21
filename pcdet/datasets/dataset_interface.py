import abc
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(abc.ABC, Dataset):
    def __init__(self, dataset_cfg, class_names: List[str], training: bool, logger=None):
        super().__init__()
        self._rng = np.random.default_rng()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.root_path = Path(dataset_cfg.DATA_PATH)
        self.logger = logger

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING, point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = (
            DataAugmentor(
                self.root_path,
                self.dataset_cfg.DATA_AUGMENTOR,
                self.class_names,
                logger=self.logger,
            )
            if self.training
            else None
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features,
            rng=self._rng,
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0

        try:
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        except AttributeError:
            self.depth_downsample_factor = None

    @abc.abstractmethod
    def evaluation(
        self, det_annos: List[Dict[str, Any]], class_names: List[str]
    ) -> Dict[str, float]:
        ...

    @property
    def mode(self):
        return "train" if self.training else "test"

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    @abc.abstractmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names) -> List[Dict[str, Any]]:
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert "gt_boxes" in data_dict, "gt_boxes should be provided for training"
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict["gt_names"]], dtype=np.bool_
            )

            data_dict = self.data_augmentor(data_dict={**data_dict, "gt_boxes_mask": gt_boxes_mask})

        if data_dict.get("gt_boxes", None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict["gt_names"], self.class_names)
            data_dict["gt_boxes"] = data_dict["gt_boxes"][selected]
            data_dict["gt_names"] = data_dict["gt_names"][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict["gt_names"]], dtype=np.int32
            )
            data_dict["gt_boxes"] = np.concatenate(
                (data_dict["gt_boxes"], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1
            )

            if data_dict.get("gt_boxes2d", None) is not None:
                data_dict["gt_boxes2d"] = data_dict["gt_boxes2d"][selected]

        if data_dict.get("points", None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict["gt_boxes"]) == 0:
            new_idx = self._rng.integers(len(self))
            return self[new_idx]

        data_dict.pop("gt_names", None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key in ["voxels", "voxel_num_points"]:
                ret[key] = np.concatenate(val)
            elif key in ["points", "voxel_coords"]:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors)
            elif key in ["gt_boxes"]:
                max_gt = max(len(x) for x in val)
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
                )
                for k in range(batch_size):
                    batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d
            elif key in ["gt_boxes2d"]:
                max_boxes = max(len(x) for x in val)
                batch_boxes2d = np.zeros(
                    (batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32
                )
                for k in range(batch_size):
                    if val[k].size > 0:
                        batch_boxes2d[k, : val[k].__len__(), :] = val[k]
                ret[key] = batch_boxes2d
            elif key in ["images", "depth_maps"]:
                # Get largest image size (H, W)
                max_h = 0
                max_w = 0
                for image in val:
                    max_h = max(max_h, image.shape[0])
                    max_w = max(max_w, image.shape[1])

                # Change size of images
                images = []
                for image in val:
                    pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                    pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                    pad_width = (pad_h, pad_w)
                    # Pad with nan, to be replaced later in the pipeline.
                    pad_value = np.nan

                    if key == "images":
                        pad_width = (pad_h, pad_w, (0, 0))
                    elif key == "depth_maps":
                        pad_width = (pad_h, pad_w)

                    image_pad = np.pad(
                        image, pad_width=pad_width, mode="constant", constant_values=pad_value
                    )

                    images.append(image_pad)
                ret[key] = np.stack(images)
            else:
                ret[key] = np.stack(val)

        ret["batch_size"] = batch_size
        return ret
