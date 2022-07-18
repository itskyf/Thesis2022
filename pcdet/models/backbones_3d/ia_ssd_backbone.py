import torch
from torch import nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules


class IASSD_Backbone(nn.Module):
    """Backbone for IA-SSD"""

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.sa_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get("AGGREGATION_MLPS", None)
        self.confidence_mlps = sa_config.get("CONFIDENCE_MLPS", None)
        self.max_translate_range = sa_config.get("MAX_TRANSLATE_RANGE", None)

        channel_out = None
        for k, nsamples in enumerate(sa_config.NSAMPLE_LIST):
            channel_in = (
                channel_out_list[self.layer_inputs[k][-1]]
                if isinstance(self.layer_inputs[k], list)
                else channel_out_list[self.layer_inputs[k]]
            )

            if self.layer_types[k] == "SALayer":
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(len(mlps)):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                aggregation_mlp = None
                if self.aggregation_mlps and len(self.aggregation_mlps[k]) > 0:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    channel_out = aggregation_mlp[-1]

                confidence_mlp = (
                    self.confidence_mlps[k].copy()
                    if self.confidence_mlps and len(self.confidence_mlps[k]) > 0
                    else None
                )

                self.sa_modules.append(
                    pointnet2_modules.PointnetSAModuleMSGSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=nsamples,
                        mlps=mlps,
                        use_xyz=True,
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class=self.num_class,
                    )
                )
            elif self.layer_types[k] == "VoteLayer":
                self.sa_modules.append(
                    pointnet2_modules.VoteLayer(
                        mlp_list=sa_config.MLPS[k],
                        pre_channel=channel_out_list[self.layer_inputs[k]],
                        max_translate_range=self.max_translate_range,
                    )
                )

            channel_out_list.append(channel_out)

        self.num_point_features = channel_out

    def split_xyz(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            points:
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict["batch_size"]
        points = batch_dict["points"]
        batch_idx, xyz, features = self.split_xyz(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size, dtype=torch.int)
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = torch.sum(batch_idx == bs_idx)

        assert torch.min(xyz_batch_cnt) == torch.max(xyz_batch_cnt)
        xyz = xyz.view(batch_size, -1, 3)
        if features is not None:
            features = features.view(batch_size, -1, features.size(-1)).transpose(1, 2).contiguous()

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        for i, sa_module in enumerate(self.sa_modules):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == "SALayer":
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred = sa_module(
                    xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz
                )
            elif self.layer_types[i] == "VoteLayer":  # i=4
                li_xyz, li_features, xyz_select, ctr_offsets = sa_module(xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)
                center_origin_batch_idx = center_origin_batch_idx[:, : centers_origin.size(1)]
                encoder_coords.append(
                    torch.cat(
                        [
                            center_origin_batch_idx[..., None].float(),
                            centers_origin.view(batch_size, -1, 3),
                        ],
                        dim=-1,
                    )
                )

            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, : li_xyz.size(1)]
            encoder_coords.append(
                torch.cat([li_batch_idx[..., None].float(), li_xyz.view(batch_size, -1, 3)], dim=-1)
            )
            encoder_features.append(li_features)
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, : li_cls_pred.size(1)]
                sa_ins_preds.append(
                    torch.cat(
                        [
                            li_cls_batch_idx[..., None].float(),
                            li_cls_pred.view(batch_size, -1, li_cls_pred.size(-1)),
                        ],
                        dim=-1,
                    )
                )
            else:
                sa_ins_preds.append([])

        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, : li_xyz.size(1)]
        ctr_batch_idx = ctr_batch_idx.flatten()
        batch_dict["ctr_offsets"] = torch.cat(
            (ctr_batch_idx[:, None].float(), ctr_offsets.reshape(-1, 3)), dim=1
        )

        batch_dict["centers"] = torch.cat(
            (ctr_batch_idx[:, None].float(), centers.reshape(-1, 3)), dim=1
        )
        batch_dict["centers_origin"] = torch.cat(
            (ctr_batch_idx[:, None].float(), centers_origin.reshape(-1, 3)), dim=1
        )

        center_features = (
            encoder_features[-1].permute(0, 2, 1).reshape(-1, encoder_features[-1].size(1))
        )
        batch_dict["centers_features"] = center_features
        batch_dict["ctr_batch_idx"] = ctr_batch_idx
        batch_dict["encoder_xyz"] = encoder_xyz
        batch_dict["encoder_coords"] = encoder_coords
        batch_dict["sa_ins_preds"] = sa_ins_preds
        batch_dict["encoder_features"] = encoder_features
        return batch_dict
