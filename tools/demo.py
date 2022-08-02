import argparse
from pathlib import Path

import numpy as np
import open3d
import torch
import torch.backends.cudnn
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from torch import nn
from tqdm.auto import tqdm
from visual_utils.open3d_vis_utils import draw_boxes

import pcdet.models
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, load_data_to_gpu
from pcdet.utils.calibration_kitti import Calibration


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
        self.sample_paths = sorted(path.glob("*.bin")) if path.is_dir() else [path]

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


def capture(vis, path: Path):
    view_control = vis.get_view_control()
    view_control.set_front([-0.75790535419768956, 0.27122179866431972, 0.59331122525009261])
    view_control.set_lookat([28.871612986983855, -6.5438185915355103, -13.066411410245985])
    view_control.set_up([0.55217956447465666, -0.21759009114810351, 0.80483059137357504])
    view_control.set_zoom(0.6)
    vis.capture_screen_image(str(path), do_render=True)


@torch.no_grad()
def main():
    args, conf = parse_config()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names: list[str] = conf.CLASS_NAMES
    data_path: Path = args.data_path
    demo_set = DemoDataset(data_path, conf.DATA_CONFIG, class_names)
    print("Number of samples:", len(demo_set))

    model_fn = getattr(pcdet.models, conf.MODEL.NAME)
    model: nn.Module = model_fn(conf.MODEL, len(class_names))
    model.to(device)
    model.load_state_dict(torch.load(args.ckpt)["model_state"])
    model.eval()

    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    if len(demo_set) > 1:
        render_option.point_size = 1.5
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
            pred_dicts = model(data_dict).pred_dicts
            pred = pred_dicts[0]

            draw_boxes(vis, pred["pred_boxes"], pred["pred_labels"])
            if not vis.poll_events():
                break
            vis.update_renderer()
    else:
        f_suffix = data_path.stem
        pic_dir = Path("./images")
        pic_dir.mkdir(exist_ok=True)

        # Raw input
        data_dict = demo_set[0]
        points = data_dict["points"]
        vis.add_geometry(axis_pcd)
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        vis.add_geometry(pts)
        capture(vis, pic_dir / f"in_{f_suffix}.png")

        # Inference
        data_dict = demo_set.collate_batch([data_dict])
        data_dict = load_data_to_gpu(data_dict)
        ctr_preds, pts_list, pred_dicts = model(data_dict, ret_points=True)
        # Convert to NumPy
        ctr_pred = ctr_preds[0].cpu().numpy()
        pts_list = [pts[0].cpu().numpy() for pts in pts_list]
        pred = pred_dicts[0]

        # Draw downsampled points
        render_option.point_size = 3
        for pts_pred in pts_list:
            num_pts = pts_pred.shape[0]
            pts.points = open3d.utility.Vector3dVector(pts_pred[:, :3])
            pts.colors = open3d.utility.Vector3dVector(np.ones((num_pts, 3)))
            vis.update_geometry(pts)
            capture(vis, pic_dir / f"points_{num_pts}_{f_suffix}.png")

        # Add centroids
        ctr = open3d.geometry.PointCloud()
        ctr.points = open3d.utility.Vector3dVector(ctr_pred[:, :3])
        ctr.colors = open3d.utility.Vector3dVector(
            np.repeat([[1, 0, 0]], ctr_pred.shape[0], axis=0)
        )
        vis.add_geometry(ctr)
        capture(vis, pic_dir / f"centers_{f_suffix}.png")
        # Draw box
        render_option.point_size = 1
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        draw_boxes(vis, pred["pred_boxes"], pred["pred_labels"])

        img_path = data_path.parent.parent / "image_2" / f"{data_path.stem}.png"
        try:  # Draw 2D box
            with Image.open(img_path) as img:
                draw = ImageDraw.Draw(img)
            calib_path = data_path.parent.parent / "calib" / f"{data_path.stem}.txt"
            calib = Calibration(calib_path)
        except FileNotFoundError:
            print("Not found image file", img_path)

        capture(vis, pic_dir / f"pred_{f_suffix}.png")
        vis.update_geometry(pts)
        capture(vis, pic_dir / f"pred_src_{f_suffix}.png")

    vis.destroy_window()
    print("Done")


def vis_points(points: NDArray[np.float32]):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    return pts


if __name__ == "__main__":
    main()
