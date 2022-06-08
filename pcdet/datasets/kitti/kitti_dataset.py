import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import ListConfig
from PIL import Image

from ...utils import box_utils, common_utils
from ..dataset_interface import IDataset, PCBatch, Prediction
from ..processor import PointFeatureEncoder
from .calibration import Calibration
from .kitti_utils import calib_to_matricies
from .object3d import get_objects_from_label


@dataclass
class KittiPCBatch(PCBatch):
    calibs: List[Calibration]

    def to(self, device: torch.device):
        return KittiPCBatch(
            self.batch_size,
            self.frame_ids,
            self.img_shapes,
            self.gt_boxes.to(device).float(),
            self.voxels.to(device).float(),
            self.voxel_coords.to(device).float(),
            self.voxel_num_points.to(device).float(),
            self.calibs,
        )


class KittiDataset(IDataset):
    def __init__(
        self,
        path: Path,
        class_names: List[str],
        info_paths: Tuple[Path, Path],
        item_names: List[str],
        fov_points_only: bool,
        augmentor_cfg: ListConfig,
        processor_cfg: ListConfig,
        pfe: PointFeatureEncoder,
        split: str,
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(class_names, augmentor_cfg, processor_cfg, pfe, training=split == "train")
        self.fov_points_only = fov_points_only
        self.item_names = item_names
        self.class_names_np = np.array(class_names)
        path = Path(path)
        self.root_split_path = path / ("training" if self.training else "testing")

        split_path = path / "ImageSets" / f"{split}.txt"
        with split_path.open() as split_file:
            self.sample_id_list = [x.strip() for x in split_file.readlines()]

        info_path = path / info_paths[self.training]
        with info_path.open("rb") as info_file:
            self.kitti_infos = pickle.load(info_file)

    def get_lidar(self, idx: str):
        lidar_path = self.root_split_path / "velodyne" / f"{idx}.bin"
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx: str):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_path = self.root_split_path / "image_2" / f"{idx}.png"
        with Image.open(img_path) as img_file:
            img = np.array(img_file, dtype=np.float32)
        return img / 255.0

    def get_image_shape(self, idx: str):
        img_path = self.root_split_path / "image_2" / f"{idx}.png"
        with Image.open(img_path) as img_file:
            img_shape = np.array([img_file.height, img_file.width], dtype=np.int32)
        return img_shape

    def get_label(self, idx: str):
        label_path = self.root_split_path / "label_2" / f"{idx}.txt"
        return get_objects_from_label(label_path)

    def get_depth_map(self, idx: str):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_path = self.root_split_path / "depth_2" / f"{idx}.png"
        with Image.open(depth_path) as depth_file:
            depth = np.array(depth_file, dtype=np.float32)
        return depth / 256.0  # TODO why 256

    def get_calib(self, idx: str):
        return Calibration(self.root_split_path / "calib" / f"{idx}.txt")

    def get_road_plane(self, idx):
        plane_path = self.root_split_path / "planes" / f"{idx}.txt"
        if not plane_path.exists():
            return None

        with plane_path.open("r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane
        return plane / np.linalg.norm(plane[0:3])

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        return np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    zeros_dict = {
        "name": np.zeros(0),
        "truncated": np.zeros(0),
        "occluded": np.zeros(0),
        "alpha": np.zeros(0),
        "bbox": np.zeros([0, 4]),
        "dimensions": np.zeros([0, 3]),
        "location": np.zeros([0, 3]),
        "rotation_y": np.zeros(0),
        "score": np.zeros(0),
        "boxes_lidar": np.zeros([0, 7]),
    }

    def gen_pred_dicts(self, pc_batch: KittiPCBatch, preds: List[Prediction]):
        def _process_sample(calib: Calibration, img_shape: npt.NDArray[np.int32], pred: Prediction):
            num_samples = pred.scores.size(0)
            if num_samples == 0:
                return KittiDataset.zeros_dict.copy()

            pred_boxes = pred.boxes.cpu().numpy()
            pred_labels = pred.labels.cpu().numpy()

            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=img_shape
            )

            return {
                "name": self.class_names_np[pred_labels - 1],
                "truncated": np.zeros(num_samples),
                "occluded": np.zeros(num_samples),
                "alpha": (
                    -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
                ),
                "bbox": pred_boxes_img,
                "boxes_lidar": pred_boxes,
                "dimensions": pred_boxes_camera[:, 3:6],
                "location": pred_boxes_camera[:, 0:3],
                "rotation_y": pred_boxes_camera[:, 6],
                "score": pred.scores.cpu().numpy(),
            }

        return [
            _process_sample(calib, img_shape, pred)
            for calib, img_shape, pred in zip(pc_batch.calibs, pc_batch.img_shapes, preds)
        ]

    def evaluation(self, det_annos):
        if "annos" not in self.kitti_infos[0].keys():
            return None, {}
        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)  # list of dict(name, alpha, bbox, etc)
        print(eval_det_annos[0])
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.kitti_infos]
        print(eval_gt_annos[0])
        raise
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            eval_gt_annos, eval_det_annos, self.class_names
        )

        return ap_result_str, ap_dict

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, index):
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info["point_cloud"]["lidar_idx"]
        img_shape = info["image"]["image_shape"]
        calib = self.get_calib(sample_idx)

        input_dict = {
            "frame_id": sample_idx,
            "calib": calib,
        }

        if "annos" in info:
            annos = info["annos"]
            annos = common_utils.drop_info_with_name(annos, name="DontCare")
            loc, dims, rots = annos["location"], annos["dimensions"], annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(
                np.float32
            )
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({"gt_names": gt_names, "gt_boxes": gt_boxes_lidar})
            if "gt_boxes2d" in self.item_names:
                input_dict["gt_boxes2d"] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict["road_plane"] = road_plane

        if "points" in self.item_names:
            points = self.get_lidar(sample_idx)
            if self.fov_points_only:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict["points"] = points
        if "images" in self.item_names:
            input_dict["images"] = self.get_image(sample_idx)

        if "depth_maps" in self.item_names:
            input_dict["depth_maps"] = self.get_depth_map(sample_idx)

        if "calib_matricies" in self.item_names:
            ret = calib_to_matricies(calib)
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = ret

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict["image_shape"] = img_shape
        return data_dict

    @staticmethod
    def collate_batch(batch_list) -> KittiPCBatch:
        pc_batch = IDataset.collate_batch(batch_list)
        return KittiPCBatch(
            pc_batch.batch_size,
            pc_batch.frame_ids,
            pc_batch.img_shapes,
            pc_batch.gt_boxes,
            pc_batch.voxels,
            pc_batch.voxel_coords,
            pc_batch.voxel_num_points,
            [sample["calib"] for sample in batch_list],
        )
