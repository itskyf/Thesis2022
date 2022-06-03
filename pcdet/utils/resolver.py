from typing import List

import numpy as np
import numpy.typing as npt


def get_grid_size(pc_range: List[float], voxel_size: List[int]) -> npt.NDArray[np.int32]:
    pc_range_np = np.array(pc_range)
    grid_size = (pc_range_np[3:6] - pc_range[0:3]) / voxel_size
    return np.round(grid_size).astype(np.int32)
