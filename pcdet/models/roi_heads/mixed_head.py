import torch
from torch import nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
# from ...ops.pointnet2.pointnet2_stack import pointnet2_stack_utils
from ...ops.pointnet2.pointnet2_stack import (
    voxel_pool_modules as voxelpool_stack_modules,
)
from ...utils import common_utils
from ..model_utils.attention_utils import TransformerEncoder, get_positional_encoder
from .roi_head_template import RoIHeadTemplate


# def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
#     """
#     Args:
#         rois: (M, 7 + C)
#         points: (N, 3)
#         sample_radius_with_roi:
#         num_max_points_of_part:
#     Returns:
#         sampled_points: (N_out, 3)
#     """
#     if points.shape[0] < num_max_points_of_part:
#         distance = (points[:, None, 0:3] - rois[None, :, 0:3]).norm(dim=-1)
#         min_dis, min_dis_roi_idx = distance.min(dim=-1)
#         roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
#         point_mask = min_dis < roi_max_dim + sample_radius_with_roi
#     else:
#         start_idx = 0
#         point_mask_list = []
#         while start_idx < points.shape[0]:
#             distance = (
#                 points[start_idx : start_idx + num_max_points_of_part, None, 0:3]
#                 - rois[None, :, 0:3]
#             ).norm(dim=-1)
#             min_dis, min_dis_roi_idx = distance.min(dim=-1)
#             roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
#             cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
#             point_mask_list.append(cur_point_mask)
#             start_idx += num_max_points_of_part
#         point_mask = torch.cat(point_mask_list, dim=0)

#     sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

#     return sampled_points, point_mask


# def sector_fps(points, num_sampled_points, num_sectors):
#     """
#     Args:
#         points: (N, 3)
#         num_sampled_points: int
#         num_sectors: int
#     Returns:
#         sampled_points: (N_out, 3)
#     """
#     sector_size = np.pi * 2 / num_sectors
#     point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
#     sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
#     points_list = []
#     xyz_batch_cnt = []
#     num_sampled_points_list = []
#     for k in range(num_sectors):
#         mask = sector_idx == k
#         cur_num_points = mask.sum().item()
#         if cur_num_points > 0:
#             points_list.append(points[mask])
#             xyz_batch_cnt.append(cur_num_points)
#             ratio = cur_num_points / points.shape[0]
#             num_sampled_points_list.append(
#                 min(cur_num_points, math.ceil(ratio * num_sampled_points))
#             )

#     if len(xyz_batch_cnt) == 0:
#         points_list.append(points)
#         xyz_batch_cnt.append(len(points))
#         num_sampled_points_list.append(num_sampled_points)
#         print(f"Warning: empty sector points detected in SectorFPS: points.shape={points.shape}")

#     points = torch.cat(point_list, dim=0)
#     xyz = point[-1, 0:3]
#     xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
#     sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

#     sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
#         xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
#     ).long()

#     sampled_points = points[sampled_pt_idxs]

#     return sampled_points


class MixedHead(RoIHeadTemplate):
    def __init__(
        self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs
    ):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        VOXEL_cfg = self.pool_cfg.VOXEL_POOL_LAYERS
        self.point_cfg = self.pool_cfg.POINT_POOL_LAYER
        self.num_points = self.point_cfg.NUM_POINTS
        # self.add_dis = self.point_cfg.ADD_DIS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # (
        #     self.point_roi_grid_pool_layer,
        #     point_c_out,
        # )
        self.point_roi_grid_pool_layer, point_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=1, config=self.point_cfg)

        voxel_c_out = 0
        self.voxel_roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = VOXEL_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=VOXEL_cfg[src_name].QUERY_RANGES,
                nsamples=VOXEL_cfg[src_name].NSAMPLE,
                radii=VOXEL_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=VOXEL_cfg[src_name].POOL_METHOD,
            )

            self.voxel_roi_grid_pool_layers.append(pool_layer)

            voxel_c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.pool_cfg.GRID_SIZE
        c_out = voxel_c_out + point_c_out
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        if self.model_cfg.get('ATTENTION', {}).get('ENABLED'):
            assert (self.model_cfg.ATTENTION.NUM_FEATURES == c_out), f"ATTENTION.NUM_FEATURES must equal voxel aggregation output dimension of {c_out}."
            pos_encoder = get_positional_encoder(self.model_cfg)
            self.attention_head = TransformerEncoder(self.model_cfg.ATTENTION, pos_encoder)

            for p in self.attention_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        shared_fc_layer = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_layer.extend(
                [
                    nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(),
                ]
            )
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_layer.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_layer)

        self.cls_layers = self.make_fc_layers(
            input_channels=self.model_cfg.SHARED_FC[-1],
            output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC,
        )

        self.reg_layers = self.make_fc_layers(
            input_channels=self.model_cfg.SHARED_FC[-1],
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC,
        )
        self.init_weights(weight_init="xavier")

    def init_weights(self, weight_init="xavier"):
        if weight_init == "kaiming":
            init_func = nn.init.kaiming_normal_
        elif weight_init == "xavier":
            init_func = nn.init.xavier_normal_
        elif weight_init == "normal":
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == "normal":
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # nn.init.normal_(self.reg_layers[-1].weight, means=0, std=0.001)

        for module_list in [self.shared_fc_layer, self.cls_layers, self.reg_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal(self.cls_layers[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.cls_layers[-1].bias, 0)
        nn.init.normal(self.reg_layers[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_layers[-1].bias, 0)

    # def sectorized_proposal_centric_sampling(self, roi_boxes, points):
    #     """
    #     Args:
    #         roi_boxes: (M, 7 + C)
    #         points: (N, 3)
    #     Returns:
    #         sampled_points: (N_out, 3)
    #     """

    #     sampled_points, _ = sample_points_with_roi(
    #         rois=roi_boxes,
    #         points=points,
    #         sample_radius_with_roi=self.point_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
    #         num_max_points_of_part=self.point_cfg.SPC_SAMPLING.get(
    #             "NUM_POINTS_OF_EACH_SAMPLE_PART", 200000
    #         ),
    #     )
    #     sampled_points = sector_fps(
    #         points=sampled_points,
    #         num_sampled_points=self.point_cfg.NUM_KEYPOINTS,
    #         num_sectors=self.point_cfg.SPC_SAMPLING.NUM_SECTORS,
    #     )
    #     return sampled_points

    # def get_sampled_points(self, batch_dict):
    #     """
    #     Args:
    #         batch_dict:
    #     Returns:
    #         keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
    #         keypoints_batch_cnt: [N1, N2, ...]
    #     """
    #     batch_size = batch_dict["batch_size"]
    #     src_points = batch_dict["points"][:, 1:]
    #     batch_indices = batch_dict["points"][:, 0].long()

    #     keypoints_list = []
    #     keypoints_batch_cnt_list = []
    #     for bs_idx in range(batch_size):
    #         bs_mask = batch_indices == bs_idx
    #         sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
    #         # if self.point_cfg.SAMPLE_METHOD == 'FPS':
    #         #     cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
    #         #         sampled_points[:, :, 0:3].contiguous(), self.point_cfg.NUM_KEYPOINTS
    #         #     ).long()

    #         #     if sampled_points.shape[1] < self.point_cfg.NUM_KEYPOINTS:
    #         #         times = int(self.point_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
    #         #         non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
    #         #         cur_pt_idxs[0] = non_empty.repeat(times)[:self.point_cfg.NUM_KEYPOINTS]

    #         #     keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

    #         # elif self.point_cfg.SAMPLE_METHOD == 'SPC':
    #         if self.point_cfg.SAMPLE_METHOD == "SPC":
    #             keypoints = self.sectorized_proposal_centric_sampling(
    #                 roi_boxes=batch_dict["rois"][bs_idx], points=sampled_points[0]
    #             )
    #             # bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
    #             # keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
    #         else:
    #             raise NotImplementedError

    #         keypoints_list.append(keypoints)
    #         keypoints_batch_cnt_list.append(keypoints.shape[0])

    #     keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
    #     keypoints_batch_cnt = torch.tensor(keypoints_batch_cnt_list, device=keypoints.device)
    #     # if len(keypoints.shape) == 3:
    #     #     batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
    #     #     keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
    #     keypoints = keypoints.view(-1, 3)
    #     return keypoints, keypoints_batch_cnt

    def roi_gird_pool(self, batch_dict):
        rois = batch_dict["rois"]
        batch_size = batch_dict["batch_size"]
        with_vf_transform = batch_dict.get("with_voxel_feature_transform", False)

        roi_grid_xyz, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )

        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        roi_grid_coords_x = torch.div(
            roi_grid_xyz[:, :, 0] - self.point_cloud_range[0],
            self.voxel_size[0],
            rounding_mode="trunc",
        )
        roi_grid_coords_y = torch.div(
            roi_grid_xyz[:, :, 1] - self.point_cloud_range[1],
            self.voxel_size[1],
            rounding_mode="trunc",
        )
        roi_grid_coords_z = torch.div(
            roi_grid_xyz[:, :, 2] - self.point_cloud_range[2],
            self.voxel_size[2],
            rounding_mode="trunc",
        )
        roi_grid_coords = torch.cat(
            [roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=1
        )

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx

        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.voxel_roi_grid_pool_layers[k]
            cur_stride = batch_dict["multi_scale_3d_strides"][src_name]
            cur_sp_tensors = batch_dict["multi_scale_3d_features"][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict["multi_scale_3d_features_post"][src_name]
            # else:
            # cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range,
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)

            cur_roi_grid_coords = torch.div(roi_grid_coords, cur_stride, rounding_mode="trunc")
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()

            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor,
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE**3, pooled_features.shape[-1]
            )
            pooled_features_list.append(pooled_features)

        voxel_pooled_feature = torch.cat(pooled_features_list, dim=1)

        num_rois = batch_dict['rois'].shape[-2]

        num_sample = self.num_points
        src = rois.new_zeros(batch_size, num_rois, num_sample, 4)
        
        for bs_idx in range(batch_size):
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, :1:5]
            cur_batch_boxes = batch_dict['rois'][bs_idx]
            cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
            dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))
            for roi_box_idx in range(0, num_rois):
                cur_roi_points = cur_points[point_mask[roi_box_idx]]

                if cur_roi_points.shape[0] >= num_sample:
                    random.seed(0)
                    index = np.random.randint(cur_roi_points.shape[0], size=num_sample)
                    cur_roi_points_sample = cur_roi_points[index]

                elif cur_roi_points.shape[0] == 0:
                    cur_roi_points_sample = cur_roi_points.new_zeros(num_sample, 4)

                else:
                    empty_num = num_sample - cur_roi_points.shape[0]
                    add_zeros = cur_roi_points.new_zeros(empty_num, 4)
                    add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                    cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                src[bs_idx, roi_box_idx, :, :] = cur_roi_points_sample

        src = src.view(batch_size * num_rois, -1, src.shape[-1])  # (b*128, 256, 4)

        # global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
        #     rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # )
        # global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)

        xyz = src[:, :, :3].view(-1, 3)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(src.shape[1] * num_rois)

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(roi_grid_points.shape[1])

        point_features = src[:, :, 3].view(-1, 1)
        
        _, point_pooled_features = self.point_roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )

        pooled_features = torch.cat([voxel_pooled_feature, point_pooled_features], dim=-1)
        return pooled_features, local_roi_grid_points

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(
            rois, batch_size_rcnn, grid_size
        )  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) - (
            local_roi_size.unsqueeze(dim=1) / 2
        )  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_positional_input(self, local_roi_grid_points):
        # TODO: Add more positional input here.
        if self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == "grid_points":
            positional_input = local_roi_grid_points
        # elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density':
        #     positional_input = points_per_part
        # elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_grid_points':
        #     positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG["TRAIN" if self.training else "TEST"]
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict["rois"] = targets_dict["rois"]
            batch_dict["roi_labels"] = targets_dict["roi_labels"]

        # RoI aware pooling
        pooled_features, local_roi_grid_points = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        if self.model_cfg.get('ATTENTION', {}).get('ENABLED'):
            src_key_padding_mask = None
            # TODO
            if self.model_cfg.ATTENTION.get("MASK_EMPTY_POINTS"):
                src_key_padding_mask = (pooled_features == 0).all(-1)

            # positional_input = self.get_positional_input(batch_dict['points'], )
            positional_input = local_roi_grid_points

            attention_ouput = self.attention_head(
                pooled_features, positional_input, src_key_padding_mask
            )

            if self.model_cfg.ATTENTION.get("COMBINE"):
                attention_ouput = pooled_features + attention_ouput
            
            pooled_features = attention_ouput
        
        # Permute
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size) # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict["batch_size"],
                rois=batch_dict["rois"],
                cls_preds=rcnn_cls,
                box_preds=rcnn_reg,
            )
            batch_dict["batch_cls_preds"] = batch_cls_preds
            batch_dict["batch_box_preds"] = batch_box_preds
            batch_dict["cls_preds_normalized"] = False
        else:
            targets_dict["rcnn_cls"] = rcnn_cls
            targets_dict["rcnn_reg"] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
