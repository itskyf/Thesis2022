from typing import List, Optional

import numpy as np


class PointFeatureEncoder:
    def __init__(
        self,
        name: str,
        used_feature_list: List[str],
        src_feature_list: List[str],
        max_sweeps: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self.used_feature_list = used_feature_list
        self.src_feature_list = src_feature_list
        self.max_sweeps = max_sweeps
        assert list(self.src_feature_list[0:3]) == ["x", "y", "z"]

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict["points"], use_lead_xyz = getattr(self, self.name)(data_dict["points"])
        data_dict["use_lead_xyz"] = use_lead_xyz

        if self.max_sweeps is not None and "timestamp" in self.src_feature_list:
            idx = self.src_feature_list.index("timestamp")
            dt = np.round(data_dict["points"][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt)) - 1, self.max_sweeps - 1)]
            data_dict["points"] = data_dict["points"][dt <= max_dt]

        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ["x", "y", "z"]:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx : idx + 1])
        point_features = np.concatenate(point_feature_list, axis=1)

        return point_features, True
