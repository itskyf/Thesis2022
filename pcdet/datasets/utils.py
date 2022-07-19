import numpy as np
import torch


def load_data_to_gpu(pcd_batch):
    for key, val in pcd_batch.items():
        if not isinstance(val, np.ndarray) or key in ["frame_id", "metadata", "calib"]:
            continue
        if key in ["image_shape"]:
            pcd_batch[key] = torch.from_numpy(val).cuda().int()
        else:
            pcd_batch[key] = torch.from_numpy(val).cuda().float()
    return pcd_batch
