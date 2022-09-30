# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

from torch import batch_norm
import torch.nn as nn
import torch 
import numpy as np
from .roi_head_template import RoIHeadTemplate
from ...ops.patch_ops import patch_ops_utils
from det3d.core import box_torch_ops
from torch.nn import functional as F
from ..registry import ROI_HEAD


class ShortcutLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels=256, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)

        self.norm1 = nn.BatchNorm1d(input_channels)
        self.norm2 = nn.BatchNorm1d(input_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x):
        """
        :param x: (B, C, N)
        :return:
            (B, C, N)
        """
        x = x + self.dropout1(x)
        x = self.norm1(x)
        x2 = self.conv2(self.dropout2(self.activation(self.conv1(x))))
        x = x + self.dropout3(x2)
        x = self.norm2(x)
        return x
    

class AttnGNNLayer(nn.Module):
    def __init__(self, input_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        mlps = model_cfg.MLPS
        self.use_feats_dist = model_cfg.USE_FEATS_DIS
        self.edge_layes = nn.ModuleList()
        in_channels = input_channels
        for i in range(len(mlps)):
            self.edge_layes.append(
                nn.Sequential(
                    nn.Conv2d(in_channels*2, mlps[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlps[i]),
                    nn.ReLU()
                )
            )
            in_channels = mlps[i]
        self.calib = nn.Sequential(
            nn.Conv1d(sum(mlps), 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1)
        )
        self.expansion = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.reduction = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.shortcut = ShortcutLayer(
            input_channels=256, hidden_channels=256, dropout=0.1
        )

    def knn(self, x, k=8):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        if idx is None:
            idx = self.knn(x)   # (batch_size, num_points, k)
        k = idx.shape[-1]
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, xyz, feats):
        B, M, K, _ = xyz.shape
        xyz = xyz.view(B * M, K, -1).permute(0, 2, 1).contiguous()
        feats = feats.view(B * M, K, -1).permute(0, 2, 1).contiguous()
        idx = self.knn(xyz) if not self.use_feats_dist else None
        x = torch.cat([xyz, feats], dim=1)
        x_list = []
        for edge_layer in self.edge_layes:
            x = self.get_graph_feature(x, idx)
            x = edge_layer(x)
            x = x.max(dim=-1)[0]
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = torch.sigmoid(self.calib(x)) * x
        x = self.expansion(x).max(dim=-1)[0].view(B, M, -1).permute(0, 2, 1)
        x = self.reduction(x)
        x = self.shortcut(x)
        return x


@ROI_HEAD.register_module
class GraphRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, pc_range=None, test_cfg=None):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size

        self.pc_range = np.array(pc_range, dtype=np.float32)
        patch_range = np.round(np.concatenate([self.pc_range[:3] - 1, self.pc_range[3:] + 1]))
        patch_size = np.array([1.0, 1.0, -1.0], dtype=np.float32)

        dfvs_config = model_cfg.DFVS_CONFIG
        self.roilocal_dfvs_pool3d_layer = patch_ops_utils.RoILocalDFVSPool3d(
            pc_range=patch_range,
            patch_size=patch_size, 
            num_dvs_points=dfvs_config.NUM_DVS_POINTS, 
            num_fps_points=dfvs_config.NUM_FPS_POINTS,
            hash_size=dfvs_config.HASH_SIZE,
            lambda_=dfvs_config.LAMBDA,
            delta=dfvs_config.DELTA,
            pool_extra_width=dfvs_config.POOL_EXTRA_WIDTH,
            num_boxes_per_patch=dfvs_config.NUM_BOXES_PER_PATCH,
        )

        self.attn_gnn_layer = AttnGNNLayer(input_channels, model_cfg.ATTN_GNN_CONFIG)

        self.shared_fc_layer = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.cls_layers = nn.Conv1d(256, self.num_class, kernel_size=1, bias=True)
        self.reg_layers = nn.Conv1d(256, code_size, kernel_size=1, bias=True)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers.weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        rois = batch_dict['rois']

        pooled_feats_local_list = []
        pooled_pts_num_list = []
        for batch_idx in range(batch_size):
            cur_points = points[batch_idx]
            ndim = 2
            pc_range = cur_points.new_tensor(self.pc_range)
            keep = torch.all((cur_points[:, :ndim] >= pc_range[:ndim]) & (cur_points[:, :ndim] <= pc_range[3:3 + ndim]), dim=-1)
            cur_points = cur_points[keep, :]
            cur_points = F.pad(cur_points, (1, 0), mode='constant', value=0).contiguous()
            cur_rois = rois[batch_idx][:, :7].unsqueeze(0).contiguous()
            pooled_pts_idx, pooled_pts_num = self.roilocal_dfvs_pool3d_layer(
                cur_points[:, :4].contiguous(),
                cur_rois
            )
            pooled_feats_local = patch_ops_utils.gather_features(cur_points[:, 1:], pooled_pts_idx, pooled_pts_num)
            pooled_feats_local_list.append(pooled_feats_local)
            pooled_pts_num_list.append(pooled_pts_num)
            
        pooled_feats_local = torch.cat(pooled_feats_local_list, dim=0)
        pooled_pts_num = torch.cat(pooled_pts_num_list, dim=0)

        # (B, M, K, 3+C)
        pooled_feats_local[..., :3] -= rois[..., :3].unsqueeze(dim=2)
        # (B*M, K, 3+C)
        pooled_feats_local = pooled_feats_local.view(-1, pooled_feats_local.shape[-2], pooled_feats_local.shape[-1])
        pooled_feats_local[..., :3] = box_torch_ops.rotate_points_along_z(
            pooled_feats_local[..., :3], -rois.view(-1, rois.shape[-1])[:, 6]
        )
        # (B*M, 8, 3)
        local_corners = box_torch_ops.corners_nd(rois[..., 3:6].view(-1, 3))
        # (B*M, K, 3+C+6)
        pooled_feats_local = torch.cat([pooled_feats_local, local_corners[:, [0, 6]].view(-1, 1, 6).repeat(1, pooled_feats_local.shape[-2], 1)], dim=-1)

        pooled_pts_num = pooled_pts_num.view(-1)  # (B*M)

        return pooled_feats_local, pooled_pts_num

    def forward(self, batch_dict, training=True):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict['batch_size'] = len(batch_dict['rois'])
        
        if training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        B, M, _ = batch_dict['rois'].shape
        roi_feats_local, roi_points_num = self.roipool3d_gpu(batch_dict)
        roi_feats_local = roi_feats_local * (roi_points_num > 0).unsqueeze(-1).unsqueeze(-1)
        roi_feats_local = roi_feats_local.view(B, M, -1, roi_feats_local.shape[-1])  # (B, M, K, C)
        roi_point_xyz = roi_feats_local[..., :3]
        roi_point_feats = roi_feats_local[..., 3:]
        # (B, C, M)
        pooled_features = self.attn_gnn_layer(roi_point_xyz, roi_point_feats)

        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().view(B * M, -1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().view(B * M, -1)  # (B, C)

        if not training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict        