import numpy as np
import torch
import torch.nn as nn


class SSFA(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super()__init__()
        self.model_cfg = model_cfg

        # Spatial block
        num_filters = self.model_cfg.NUM_FILTERS
        layer_strides = self.model_cfg.LAYER_STRIDES
        kernel_sizes = self.model_cfg.KERNEL_SIZES

        # Semantic block
        seg_layer_strides = self.model_cfg.SEG_LAYER_STRIDES
        seg_kernel_sizes = self.model_cfg.SEG_KERNEL_SIZES
        seg_num_filters = self.model_cfg.SEG_NUM_FILTERS
        
        # Spatial block 2
        trans_num_filters = self.model_cfg.TRANS_NUM_FILTERS
        trans_layer_strides = self.model_cfg.TRANS_LAYER_STRIDES
        trans_kernel_sizes = self.model_cfg.TRANS_KERNEL_SIZES

        # Semantic deconv flow to fuse with spatial
        deconv_layer_filters_0 = self.model_cfg.DECONV_LAYER_FILTERS_0
        deconv_kernel_sizes_0 = self.model_cfg.DECONV_KERNEL_SIZES_0
        deconv_strides_0 = self.model_cfg.DECONV_STRIDES_0

        # Segment deconv flow
        deconv_layer_filters_1 = self.model_cfg.DECONV_LAYER_FILTERS_1
        deconv_kernel_sizes_1 = self.model_cfg.DECONV_KERNEL_SIZES_1
        deconv_strides_1 = self.model_cfg.DECONV_STRIDES_1

        # Weight spatial for fusion
        conv_0_layer_filters = self.model_cfg.CONV_0_LAYER_FILTERS
        conv_0_kernel_sizes = self.model_cfg.CONV_0_KERNEL_SIZERS
        conv_0_strides = self.model_cfg.CONV_0_STRIDES
        w_0_layer_filters = self.model_cfg.W_0_LAYER_FILTERS
        w_0_kernel_sizes = self.model_cfg.W_0_KERNEL_SIZES
        w_0_strides = self.model_cfg.W_0_STRIDES

        # Weight semantic for fustion
        conv_1_layer_filters = self.model_cfg.CONV_1_LAYER_FILTERS
        conv_1_kernel_sizes = self.model_cfg.CONV_1_KERNEL_SIZERS
        conv_1_strides = self.model_cfg.CONV_1_STRIDES
        w_1_layer_filters = self.model_cfg.W_1_LAYER_FILTERS
        w_1_kernel_sizes = self.model_cfg.W_1_KERNEL_SIZES
        w_1_strides = self.model_cfg.W_1_STRIDES

        self.bottom_up_block_0 = nn.Sequential()
        num_filters = [input_channels, *num_filters]
        for i in range(len(layer_strides)):
            padding = 1 if kernel_sizes[i] == 3 else 0
            self.bottom_up_block_0.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=num_filters[i], out_channels=num_filters[i+1], kernel_size=kernel_sizes[i], stride=layer_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(num_filters[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        
        self.bottom_up_block_1 = nn.Sequential()
        seg_num_filters = [num_filters[-1], *seg_num_filters]
        for i in range(len(seg_layer_strides)):
            padding = 1 if seg_kernel_sizes[i] == 3 else 0
            self.bottom_up_block_1.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=seg_num_filters[i], out_channels=seg_num_filters[i+1], kernel_size=seg_kernel_sizes[i], padding=padding, stride=seg_layer_strides[i], bias=False),
                nn.BatchNorm2d(seg_num_filters[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        
        self.trans = nn.Sequential()
        trans_num_filters = [num_filters[-1], *trans_num_filters]
        for i in range(len(trans_layer_strides)):
            padding = 1 if trans_kernel_sizes[i] == 3 else 0
            self.trans.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=trans_num_filters[i], out_channels=trans_num_filters[i+1], kernel_size=trans_kernel_sizes[i], stride=trans_layer_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(trans_layer_strides[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        
        self.deconv_block_0 = nn.Sequential()
        deconv_layer_filters_0 = [seg_num_filters[-1], *deconv_layer_filters_0]
        for i in range(len(deconv_strides_0)):
            padding = 1 if deconv_kernel_sizes_0[i] == 3 else 0
            self.deconv_block_0.add_module(f'deconv_{i}', nn.Sequential(
                nn.ConvTranspose2d(in_channels=deconv_layer_filters_0[i], out_channels=deconv_layer_filters_0[i+1], kernel_size=deconv_kernel_sizes_0[i], stride=deconv_strides_0[i], padding=padding, bias=False),
                nn.BatchNorm2d(deconv_layer_filters_0[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        
        self.deconv_block_1 = nn.Sequential()
        deconv_layer_filters_1 = [seg_num_filters[-1], *deconv_layer_filters_1]
        for i in range(len(deconv_strides_0)):
            padding = 1 if deconv_kernel_sizes_1[i] == 3 else 0
            self.deconv_block_1.add_module(f'deconv_{i}', nn.Sequential(
                nn.ConvTranspose2d(in_channels=deconv_layer_filters_1[i], out_channels=deconv_layer_filters_1[i+1], kernel_size=deconv_kernel_sizes_1[i], stride=deconv_strides_1[i], padding=padding, bias=False),
                nn.BatchNorm2d(deconv_layer_filters_1[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.conv_0 = nn.Sequential()
        conv_0_layer_filters = [deconv_layer_filters_0[-1], *conv_0_layer_filters]
        for i in range(len(conv_0_strides)):
            padding = 1 if conv_0_kernel_sizes[i]==3 else 0
            self.conv_0.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=conv_0_layer_filters[0], out_channels=conv_0_layer_filters[i+1], kernel_size=conv_0_kernel_sizes[i], stride=conv_0_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(conv_0_layer_filters[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))
        
        self.w_0 = nn.Sequential()
        w_0_layer_filters = [conv_0_layer_filters[-1], *w_0_layer_filters]
        for i in range(len(w_0_strides)):
            padding = 1 if w_0_kernel_sizes[i]==3 else 0
            self.w_0.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=w_0_layer_filters[i], out_channels=w_0_layer_filters[i+1], kernel_size=w_0_kernel_sizes[i], stride=w_0_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(w_0_layer_filters[i+1])
            ))
        
        self.conv_1 = nn.Sequential()
        conv_1_layer_filters = [deconv_layer_filters_0[-1], *conv_1_layer_filters]
        for i in range(len(conv_1_strides)):
            padding = 1 if conv_1_kernel_sizes[i]==3 else 0
            self.conv_1.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=conv_1_layer_filters[0], out_channels=conv_1_layer_filters[i+1], kernel_size=conv_0_kernel_sizes[i], stride=conv_0_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(conv_1_layer_filters[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))
        
        self.w_1 = nn.Sequential()
        w_1_layer_filters = [conv_1_layer_filters[-1], *w_1_layer_filters]
        for i in range(len(w_1_strides)):
            padding = 1 if w_1_kernel_sizes[i]==3 else 0
            self.w_1.add_module(f'conv_{i}', nn.Sequential(
                nn.Conv2d(in_channels=w_0_layer_filters[i], out_channels=w_0_layer_filters[i+1], kernel_size=w_0_kernel_sizes[i], stride=w_0_strides[i], padding=padding, bias=False),
                nn.BatchNorm2d(w_0_layer_filters[i+1])
            ))

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        # ret_dict = {}
        x_0 = spatial_features
        x_0 = self.bottom_up_block_0(x_0)
        x_1 = self.bottom_up_block_1(x_0)
        x_0 = self.trans(x_0)
        x_0 = self.deconv_block_0(x_1) + x_0
        x_1 = self.deconv_block_1(x_1)
        x_0 = self.conv_0(x_0)
        x_1 = self.conv_1(x_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_0 * x_weight[:, 0:1, :, :] + x_1 * x_weight[:, 1:, :, :]

        data_dict['spatial_features_2d'] = x_output
        return data_dict