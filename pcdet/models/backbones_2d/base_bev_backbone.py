import itertools
from typing import List, Optional

import torch
from torch import nn


class BaseBEVBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layer_nums: List[int],
        layer_strides: List[int],
        num_filters: List[int],
        num_upsample_filters: Optional[List[int]] = None,
        upsample_strides: Optional[List[int]] = None,
    ):
        super().__init__()
        assert len(layer_nums) == len(layer_strides)
        assert len(layer_nums) == len(num_filters)

        if num_upsample_filters and upsample_strides:
            assert len(num_upsample_filters) == len(upsample_strides)
        else:
            upsample_strides = num_upsample_filters = []

        channels = [in_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for channel, ln, ls, nf in zip(channels, layer_nums, layer_strides, num_filters):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(channel, nf, 3, ls, padding=0, bias=False),
                nn.BatchNorm2d(nf, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(ln):
                cur_layers.extend(
                    [
                        nn.Conv2d(nf, nf, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(nf, eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.blocks.append(nn.Sequential(*cur_layers))
        # blocks = [
        #    nn.Sequential(
        #        nn.ZeroPad2d(1),
        #        nn.Conv2d(channel, num_filter, 3, stride, 0, bias=False),
        #        nn.BatchNorm2d(num_filter, eps=1e-3, momentum=0.01),
        #        nn.ReLU(),
        #        *itertools.chain.from_iterable(
        #            itertools.repeat(
        #                [
        #                    nn.Conv2d(num_filter, num_filter, 3, padding=1, bias=False),
        #                    nn.BatchNorm2d(num_filter, eps=1e-3, momentum=0.01),
        #                    nn.ReLU(),
        #                ],
        #                layer_num,
        #            )
        #        )
        #    )
        #    for channel, num_filter, stride, layer_num in zip(
        #        channels, num_filters, layer_strides, layer_nums
        #    )
        # ]
        # self.blocks = nn.ModuleList(blocks)
        # TODO walrus for python 3.8
        kernel_sizes = [round(1 / stride) for stride in upsample_strides]
        blocks = [
            nn.Sequential(
                nn.ConvTranspose2d(num_filter, num_upsample_filter, stride, stride, bias=False)
                if stride >= 1
                else nn.Conv2d(
                    num_filter, num_upsample_filter, kernel_size, stride=kernel_size, bias=False
                ),
                nn.BatchNorm2d(num_upsample_filter, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
            for num_filter, num_upsample_filter, kernel_size, stride in zip(
                num_filters, num_upsample_filters, kernel_sizes, upsample_strides
            )
        ]

        if len(num_upsample_filters) > len(num_filters):
            c_in = sum(num_upsample_filters)
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )
        self.deblocks = nn.ModuleList(blocks)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []
        # TODO list comprehension?
        for block, deblock in itertools.zip_longest(self.blocks, self.deblocks):
            if block is None:
                break
            spatial_features = block(spatial_features)
            ups.append(deblock(spatial_features) if deblock else spatial_features)

        if len(ups) > 0:
            spatial_features = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        if len(self.deblocks) > len(self.blocks):
            spatial_features = self.deblocks[-1](spatial_features)

        return spatial_features
