import enum
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import spconv.pytorch as spconv
import torch
from torch import nn


@dataclass
class MultiScale3DFeatures:
    stride1: int
    stride2: int
    stride3: int
    stride4: int
    x_conv1: spconv.SparseConvTensor
    x_conv2: spconv.SparseConvTensor
    x_conv3: spconv.SparseConvTensor
    x_conv4: spconv.SparseConvTensor


@dataclass
class SparseBackboneOut:
    sparse_out: spconv.SparseConvTensor
    sparse_stride: int
    multiscale_3d_features: MultiScale3DFeatures


class ConvType(enum.Enum):
    SUBM = enum.auto()
    SPCONV = enum.auto()
    INVERSECONV = enum.auto()


def post_act_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: Union[int, Tuple[int, ...]],
    norm_fn: Callable[[int], nn.Module] = nn.BatchNorm1d,
    indice_key: Optional[str] = None,
    conv_type=ConvType.SUBM,
):
    if conv_type is ConvType.SUBM:
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    elif conv_type is ConvType.SPCONV:
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type is ConvType.INVERSECONV:
        conv = spconv.SparseInverseConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    else:
        raise NotImplementedError
    return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride, norm_fn, downsample=None, indice_key=None):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, 3, stride, padding=1, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(planes, planes, 3, stride, padding=1, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out: spconv.SparseConvTensor = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        return out.replace_feature(self.relu(out.features))


class VoxelBackbone8x(nn.Module):
    # TODO use VoxelBackbone8x.dims in __init__
    dims = (16, 32, 64, 64)

    def __init__(
        self,
        in_channels: int,
        grid_size: npt.NDArray[np.int32],
        last_pad: int = 0,
    ):
        super().__init__()
        batchnorm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key="subm1"),
            batchnorm1d_fn(16),
            nn.ReLU(),
        )

        self.conv1 = spconv.SparseSequential(
            post_act_block(16, 16, 3, 1, 1, batchnorm1d_fn, "subm1")
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            post_act_block(16, 32, 3, 2, 1, batchnorm1d_fn, "spconv2", ConvType.SPCONV),
            post_act_block(32, 32, 3, 1, 1, batchnorm1d_fn, "subm2"),
            post_act_block(32, 32, 3, 1, 1, batchnorm1d_fn, "subm2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            post_act_block(32, 64, 3, 2, 1, batchnorm1d_fn, "spconv3", ConvType.SPCONV),
            post_act_block(64, 64, 3, 1, 1, batchnorm1d_fn, "subm3"),
            post_act_block(64, 64, 3, 1, 1, batchnorm1d_fn, "subm3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            post_act_block(64, 64, 3, 2, (0, 1, 1), batchnorm1d_fn, "spconv4", ConvType.SPCONV),
            post_act_block(64, 64, 3, 1, 1, batchnorm1d_fn, "subm4"),
            post_act_block(64, 64, 3, 1, 1, batchnorm1d_fn, "subm4"),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                64, 128, (3, 1, 1), (2, 1, 1), last_pad, bias=False, indice_key="spconv_down2"
            ),
            batchnorm1d_fn(128),
            nn.ReLU(),
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        in_sp_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(), self.sparse_shape, batch_size
        )
        x = self.conv_input(in_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        sparse_out = self.conv_out(x_conv4)
        multiscale_3d_features = MultiScale3DFeatures(
            1, 2, 4, 8, x_conv1, x_conv2, x_conv3, x_conv4
        )
        return SparseBackboneOut(sparse_out, 8, multiscale_3d_features)


class VoxelResBackbone8x(nn.Module):
    def __init__(self, in_channels, grid_size, last_pad: int = 0):
        super().__init__()
        batchnorm1d_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key="subm1"),
            batchnorm1d_fn(16),
            nn.ReLU(),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, 1, norm_fn=batchnorm1d_fn, indice_key="res1"),
            SparseBasicBlock(16, 16, 1, norm_fn=batchnorm1d_fn, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            post_act_block(16, 32, 3, 2, 1, batchnorm1d_fn, "spconv2", ConvType.SPCONV),
            SparseBasicBlock(32, 32, 1, norm_fn=batchnorm1d_fn, indice_key="res2"),
            SparseBasicBlock(32, 32, 1, norm_fn=batchnorm1d_fn, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            post_act_block(32, 64, 3, 2, 1, batchnorm1d_fn, "spconv3", ConvType.SPCONV),
            SparseBasicBlock(64, 64, 1, norm_fn=batchnorm1d_fn, indice_key="res3"),
            SparseBasicBlock(64, 64, 1, norm_fn=batchnorm1d_fn, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            post_act_block(64, 128, 3, 2, (0, 1, 1), batchnorm1d_fn, "spconv4", ConvType.SPCONV),
            SparseBasicBlock(128, 128, 1, norm_fn=batchnorm1d_fn, indice_key="res4"),
            SparseBasicBlock(128, 128, 1, norm_fn=batchnorm1d_fn, indice_key="res4"),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), last_pad, bias=False, indice_key="spconv_down2"
            ),
            batchnorm1d_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {"x_conv1": 16, "x_conv2": 32, "x_conv3": 64, "x_conv4": 128}

    def forward(self, info):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=info.voxel_features,
            indices=info.voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=info.batch_size,
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        info.encoded_spconv_tensor = out
        info.encoded_spconv_tensor_stride = 8

        info.multi_scale_3d_features.x_conv1 = x_conv1
        info.multi_scale_3d_features.x_conv2 = x_conv2
        info.multi_scale_3d_features.x_conv3 = x_conv3
        info.multi_scale_3d_features.x_conv4 = x_conv4
        info.multi_scale_3d_features.stride1 = 1
        info.multi_scale_3d_features.stride2 = 2
        info.multi_scale_3d_features.stride3 = 4
        info.multi_scale_3d_features.stride4 = 8
        return info
