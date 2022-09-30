import torch
from torch.autograd import Function
import torch.nn as nn
from typing import List
from . import patch_ops_cuda
import numpy as np
from det3d.core.bbox import box_torch_ops


def gather_features(feats, pooled_pts_idx, pooled_pts_num):
    """
    Args:
        feats: (N1+N2+..., C)
        pooled_pts_idx: (..., K)
        pooled_pts_num: (..., )
    Returns:
        pooled_features: (..., K, C)
    """
    pooled_feats = feats.new_zeros(*pooled_pts_idx.shape, feats.shape[-1])
    # (T, K, C), (T, K)
    pooled_feats[pooled_pts_num > 0] = feats[pooled_pts_idx[pooled_pts_num > 0].long()]
    return pooled_feats


def generate_points2voxels(points, pc_range, voxel_size, to_pillars=False):
    """
    Args:
        points: (N1 + N2 +..., 4) [batch_idx, x, y, z], preassume points are in pc_range
        pc_range: (6,), tensor type
        voxel_size: (3,), tensor type
    Return:
        unique_coords: (M1 + M2 + ..., 3 or 4), [batch_idx, Y, X] or [batch_idx, Z, Y, X]
        inverse_indices: (N1 + N2 + ...,), stacked index of unique_coords
    """
    ndim = 2 if to_pillars else 3

    coords = ((points[:, 1:1 + ndim] - pc_range[:ndim]) / voxel_size[:ndim]).to(torch.int64)
    coords = torch.cat([points[:, 0:1].long(), torch.flip(coords, dims=[-1])], dim=-1)
    unique_coords, inverse_indices = coords.unique(sorted=False, return_inverse=True, dim=0)
    return unique_coords, inverse_indices


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxels2pinds(indices, spatial_shape, batch_size):
    """
    Args:
        indices: (N1 + N2 + ..., 3 or 4) [batch_idx, Z, Y, X] or [batch_idx, Y, X]
        spatial_shape: [Z, Y, X] or [Y, X], numpy type
    Return:
        v2pinds_tensor: (B, Z, Y, X) or (B, Y, X), stacked index of indices
    """
    point_indices = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


class PatchQuery(Function):

    @staticmethod
    def forward(ctx, boxes3d: torch.Tensor, patch_indices: torch.Tensor, patches: torch.Tensor, 
        offset_x: float, offset_y: float, patch_size_x: float, patch_size_y: float, num_boxes_per_patch: int):
        """
        Args:
            ctx:
            boxes3d: (B, M, 7)
            patch_indices: (B, Y, X) recordes the point indices of patches
            patches: (N1 + N2 + ..., 3) [batch_id, y, x]
        Returns:
            patch2box_indices: (N1 + N2 + ..., num_boxes_per_patch + 1)
        """
        assert boxes3d.is_contiguous()
        assert patch_indices.is_contiguous()

        M = boxes3d.shape[1]
        B, Y, X = patch_indices.shape
        N = patches.shape[0]
        patch2box_indices = torch.cuda.IntTensor(N, num_boxes_per_patch + 1).zero_()
        patch_ops_cuda.patch_query_wrapper(B, M, num_boxes_per_patch, X, Y, offset_x, offset_y, 
            patch_size_x, patch_size_y, boxes3d, patch_indices, patch2box_indices)

        return patch2box_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


patch_query = PatchQuery.apply


class RoILocalDFVSPool3d(nn.Module):
    def __init__(self, pc_range, patch_size, num_dvs_points=1024, num_fps_points=256, hash_size=4099, lambda_=0.22, delta=70, pool_extra_width=1.0, num_boxes_per_patch=16):
        super().__init__()
        self.pc_range = pc_range
        self.patch_size = patch_size
        self.num_dvs_points = num_dvs_points
        self.num_fps_points = num_fps_points
        self.pool_extra_width = pool_extra_width
        self.num_boxes_per_patch = num_boxes_per_patch
        self.hash_size = hash_size
        self.lambda_ = lambda_
        self.delta = delta

    def forward(self, points, boxes3d):
        """
        Args:
            points: (N1 + N2 + ..., 4) [batch_idx, x, y, z]
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]
        Returns:
            pooled_pts_idx: (B, M, 512)
            pooled_pts_num: (B, M)
        """
        assert points.shape.__len__() == 2 and points.shape[-1] == 4
        boxes3d = box_torch_ops.to_pcdet(boxes3d)
        batch_size = boxes3d.shape[0]
        pc_range = boxes3d.new_tensor(self.pc_range)
        patch_size = boxes3d.new_tensor(self.patch_size)
        shape_np = np.round((self.pc_range[3:] - self.pc_range[:3]) / self.patch_size).astype(np.int32)
        # patches: (K1 + K2 + ..., 3) [batch_id, y, x]
        # patch_indices: (B, Y, X)
        # point2patch_indices: (N1 + N2 + ..., )
        patches, point2patch_indices = generate_points2voxels(points, pc_range, patch_size, to_pillars=True)
        patch_indices = generate_voxels2pinds(patches, shape_np[:2][::-1], batch_size)

        pooled_boxes3d = box_torch_ops.enlarge_box3d(boxes3d.view(-1, 7), self.pool_extra_width).view(batch_size, -1, 7)
        # (K1 + K2 + ..., num_boxes_per_patch + 1)
        patch2box_indices = patch_query(pooled_boxes3d.contiguous(), patch_indices.contiguous(), patches, 
            self.pc_range[0], self.pc_range[1], self.patch_size[0], self.patch_size[1], self.num_boxes_per_patch)
        return RoILocalDFVSPool3dFunction.apply(
            points[:, 1:], pooled_boxes3d, point2patch_indices.int(), patch2box_indices, self.num_dvs_points, self.num_fps_points, self.hash_size, self.lambda_, self.delta, self.num_boxes_per_patch
        )


class RoILocalDFVSPool3dFunction(Function):
    @staticmethod
    def forward(ctx, points, boxes3d, point2patch_indices, patch2box_indices, num_dvs_points, num_fps_points, hash_size, lambda_, delta, num_boxes_per_patch):
        """
        Args:
            ctx:
            points: (N1 + N2 + ..., 3)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]
            point2patch_indices: (N1 + N2 + ..., )
            patch2box_indices: (K1 + K2 + ..., num_boxes_per_patch + 1)

        Returns:
            init_pooled_pts_idx: (B, M, 4096)
            pooled_pts_idx: (B, M, 512)
            pooled_pts_num: (B, M)
        """
        assert points.shape.__len__() == 2 and points.shape[-1] == 3
        assert points.shape[0] == point2patch_indices.shape[0]
        assert boxes3d.shape.__len__() == 3 and boxes3d.shape[-1] == 7
        batch_size, boxes_num = boxes3d.shape[:2]
        pts_num, _ = points.shape

        pooled_pts_idx = points.new_zeros((batch_size, boxes_num, num_fps_points)).int()
        pooled_pts_num = points.new_zeros((batch_size, boxes_num)).int()

        patch_ops_cuda.roilocal_dfvs_pool3d_wrapper(
            batch_size, pts_num, boxes_num, num_dvs_points, num_fps_points, num_boxes_per_patch, hash_size, lambda_, delta, 
            points.contiguous(), boxes3d.contiguous(), point2patch_indices.contiguous(), patch2box_indices.contiguous(), 
            pooled_pts_num, pooled_pts_idx
        )

        return pooled_pts_idx, pooled_pts_num

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
