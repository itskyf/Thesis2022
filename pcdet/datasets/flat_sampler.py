import torch
from torch.utils.data import DistributedSampler

from .dataset_interface import DatasetTemplate


class FlatDistSampler(DistributedSampler):
    def __init__(self, dataset: DatasetTemplate):
        super().__init__(dataset)

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
