import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor import DataProcessor, PointFeatureEncoder


@dataclass
class PointCloud:
    batch_size: int
    gt_boxes: torch.Tensor
    voxels: torch.Tensor
    voxel_coords: torch.Tensor
    voxel_num_points: torch.Tensor

    def cuda(self):
        return PointCloud(
            self.batch_size,
            self.gt_boxes.cuda().float(),
            self.voxels.cuda().float(),
            self.voxel_coords.cuda().float(),
            self.voxel_num_points.cuda().float(),
        )


class IDataset(abc.ABC, Dataset):
    def __init__(
        self,
        class_names: List[str],
        augmentor_cfg: ListConfig,
        processor_cfg: ListConfig,
        pfe: PointFeatureEncoder,
        training: bool,
    ):
        super().__init__()
        # TODO refactor augmentor to hydra instantiate
        self.augmentor = DataAugmentor(augmentor_cfg) if training else None
        self.data_processor = DataProcessor(processor_cfg, training)
        self.pfe = pfe

        self.class_names = class_names
        self.training = training
        try:
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        except AttributeError:
            self.depth_downsample_factor = None

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    @abc.abstractmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path: Optional[Path]):
        """
        To support a custom dataset, implement this function
        to receive the predicted results from the model
        then transform the unified normative coordinate to your required coordinate
        and optionally save them to disk.

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
        if self.augmentor is not None:
            assert "gt_boxes" in data_dict, "gt_boxes should be provided for training"
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict["gt_names"]], dtype=bool
            )
            data_dict = self.augmentor(data_dict={**data_dict, "gt_boxes_mask": gt_boxes_mask})

        if data_dict.get("gt_boxes", None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict["gt_names"], self.class_names)
            data_dict["gt_boxes"] = data_dict["gt_boxes"][selected]
            data_dict["gt_names"] = data_dict["gt_names"][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict["gt_names"]], dtype=np.int32
            )
            gt_boxes = np.concatenate(
                (data_dict["gt_boxes"], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1
            )
            data_dict["gt_boxes"] = gt_boxes

            if data_dict.get("gt_boxes2d", None) is not None:
                data_dict["gt_boxes2d"] = data_dict["gt_boxes2d"][selected]

        if data_dict.get("points", None) is not None:
            data_dict = self.pfe(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict["gt_boxes"]) == 0:
            new_index = np.random.randint(len(self))
            return self.__getitem__(new_index)

        data_dict.pop("gt_names", None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list) -> PointCloud:
        batch_size = len(batch_list)
        voxels = torch.from_numpy(np.concatenate([sample["voxels"] for sample in batch_list]))
        voxel_num_points = torch.from_numpy(
            np.concatenate([sample["voxel_num_points"] for sample in batch_list])
        )
        voxel_coords = [
            np.pad(sample["voxel_coords"], ((0, 0), (1, 0)), mode="constant", constant_values=i)
            for i, sample in enumerate(batch_list)
        ]
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        gt_list = [sample["gt_boxes"] for sample in batch_list]
        max_gt = max([len(x) for x in gt_list])
        last_dim = gt_list[0].shape[-1]  # TODO better naming
        batch_gt_boxes3d = np.zeros((batch_size, max_gt, last_dim), dtype=np.float32)
        for k, gt in enumerate(gt_list):
            batch_gt_boxes3d[k, : len(gt), :] = gt
        gt_boxes = torch.from_numpy(batch_gt_boxes3d)

        return PointCloud(batch_size, gt_boxes, voxels, voxel_coords, voxel_num_points)
