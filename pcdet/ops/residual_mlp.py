import torch
import torch.nn as nn
import torch.nn.functional as functional

class ResPBlock2D(nn.Module):
    def __init__(self, channels, res_expansion=1, kernel_size=1, bias=False):
        super(ResPBlock2D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*res_expansion, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(channels*res_expansion),
            nn.GELU(),
            nn.Conv2d(in_channels=channels*res_expansion, out_channels=channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return nn.GELU(self.mlp(x)+x)

class ResPBlock1D(nn.Module):
    def __init__(self, channels, res_expansion=1, kernel_size=1, bias=False):
        super(ResPBlock1D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels*res_expansion, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channels*res_expansion),
            nn.GELU(),
            nn.Conv1d(in_channels=channels*res_expansion, out_channels=channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channels),
        )
        # self.mlp2 = nn.Sequential(
        #     # nn.ReLU()
        # )
    def forward(self, x):
        return nn.GELU(self.mlp(x)+x)