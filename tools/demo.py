import argparse
from pathlib import Path

import numpy as np
import open3d
import torch
import torch.backends.cudnn
from numpy.typing import NDArray
from torch import nn
from tqdm.auto import tqdm
from visual_utils.open3d_vis_utils import draw_boxes

import pcdet.models
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, load_data_to_gpu


class DemoDataset(DatasetTemplate):
    def __init__(self, path: Path, dataset_cfg, class_names: list[str]):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(dataset_cfg, class_names, training=False)
        self.sample_paths = sorted(path.glob("*.bin"))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index: int):
        points = np.fromfile(self.sample_paths[index], dtype=np.float32)
        input_dict = {"points": points.reshape(-1, 4), "frame_id": index}
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos):
        ...

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts):
        ...


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("data_path", type=Path)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_path, cfg)
    return args, cfg


@torch.no_grad()
def main():
    args, conf = parse_config()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names: list[str] = conf.CLASS_NAMES
    demo_set = DemoDataset(args.data_path, conf.DATA_CONFIG, class_names)
    print("Number of samples:", len(demo_set))

    model_fn = getattr(pcdet.models, conf.MODEL.NAME)
    model: nn.Module = model_fn(conf.MODEL, len(class_names))
    model.to(device)
    model.load_state_dict(torch.load(args.ckpt)["model_state"])
    model.eval()

    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    for data_dict in tqdm(demo_set):
        vis.clear_geometries()
        vis.add_geometry(axis_pcd)

        points = data_dict["points"]
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        vis.add_geometry(pts)

        data_dict = demo_set.collate_batch([data_dict])
        data_dict = load_data_to_gpu(data_dict)
        pred_dicts = model.forward(data_dict)[0].pred_dicts
        pred = pred_dicts[0]

        draw_boxes(vis, pred["pred_boxes"], pred["pred_labels"])
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    print("Done")


def vis_points(points: NDArray[np.float32]):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    return pts


if __name__ == "__main__":
    main()
