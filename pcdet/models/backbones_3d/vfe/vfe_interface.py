import abc

from torch import nn


class IVoxelFE(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def output_feature_dim(self) -> int:
        ...
