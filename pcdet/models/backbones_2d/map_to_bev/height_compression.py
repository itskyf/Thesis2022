import spconv.pytorch as spconv
from torch import nn


class HeightCompression(nn.Module):
    def forward(self, sparse_out: spconv.SparseConvTensor):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        spatial_features = sparse_out.dense()
        n, c, d, h, w = spatial_features.shape
        return spatial_features.view(n, c * d, h, w)
