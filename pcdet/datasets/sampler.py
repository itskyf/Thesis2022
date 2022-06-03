from typing import Optional

import torch
from torch.utils.data import Dataset, DistributedSampler


class FlatDistSampler(DistributedSampler):
    def __init__(
        self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None
    ):
        super().__init__(dataset, num_replicas, rank)

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
