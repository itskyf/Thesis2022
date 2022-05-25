from collections import namedtuple

import kornia
import numpy as np
import torch

from .detectors import (
    CT3D,
    CT3D3C,
    PVRCNN,
    CaDDN,
    CenterPoint,
    PartA2Net,
    PointPillar,
    PointRCNN,
    PVRCNNPlusPlus,
    SECONDNet,
    SECONDNetIoU,
    VoTrRCNN,
    VoTrSSD,
    VoxelRCNN,
)

__all__ = [
    "CaDDN",
    "CenterPoint",
    "CT3D",
    "CT3D3C",
    "PartA2Net",
    "PointPillar",
    "PointRCNN",
    "PVRCNN",
    "PVRCNNPlusPlus",
    "SECONDNet",
    "SECONDNetIoU",
    "VoTrRCNN",
    "VoTrSSD",
    "VoxelRCNN",
]


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray) or key in ["frame_id", "metadata", "calib"]:
            continue
        elif key in ["images"]:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ["image_shape"]:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict["loss"].mean()
        if hasattr(model, "update_global_step"):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
