import abc
from typing import List, Tuple

import torch


class ITargetAssigner(abc.ABC):
    @abc.abstractmethod
    def assign_targets(
        self, all_anchors: List[torch.Tensor], gt_boxes_with_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
