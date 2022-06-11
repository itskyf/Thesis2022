from torch import nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules
from ...utils import common_utils
from ..model_utils.attention_utils import TransformerEncoder, get_positional_encoder
from .roi_head_template import RoIHeadTemplate

class HopelessHead(RoIHeadTemplate):
    def __init__(self, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.num_points = model_cfg.NUM_POINTS
        self.roi_grid_pool_layer, num_c_out = pointnet2_modules.build_local_aggregation_module(
            input_channels=1, config=self.model_cfg.ROI_GRID_POOL
        )

        if self.model_cfg.get('ATTENTION', {}).get('ENABLED'):
            assert (self.model_cfg.ATTENTION.NUM_FEATURES == num_c_out), f"ATTENTION.NUM_FEATURES must equal voxel aggregation output dimension of {num_c_out}."
            pos_encoder = get_positional_encoder(self.model_cfg)
            self.attention_head = TransformerEncoder(self.model_cfg.ATTENTION, pos_encoder)

            for p in self.attention_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend(
                [
                    nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(),
                ]
            )
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC,
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
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
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
        Returns:

        """
        batch_size = batch_dict["batch_size"]
        rois = batch_dict["rois"]
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

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )

        xyz = src[:, :4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = src[:, 0]

        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])

        point_features = src[:, 4]
        
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE**3, pooled_features.shape[-1]
        )

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
        if self.model_cfg.ATTENTION.POSITIONAL_ENCODER == "grid_points":
            positional_input = local_roi_grid_points
        # elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density':
        #     positional_input = points_per_part
        # elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_grid_points':
        #     positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG["TRAIN" if self.training else "TEST"]
        )
        if self.training:
            targets_dict = batch_dict.get("roi_targets_dict", None)
            if targets_dict is None:
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