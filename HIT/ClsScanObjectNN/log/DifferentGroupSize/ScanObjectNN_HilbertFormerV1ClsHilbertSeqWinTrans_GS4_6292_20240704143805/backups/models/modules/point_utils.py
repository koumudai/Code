import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable
Tensor = torch.Tensor
import pointcloud_ops.pointcloud_utils as pcops


# ---------------------------------------- PointCloud Operations ---------------------------------------- #
def furthest_point_sampling(pts_all: Tensor, num_spl: int) -> Tensor:
    '''
    Description:
        furthest point sampling.
    Input:
        pts_all: the coordinates of all points. Shape: (batch_size, num_pts, dim_pts).
        num_spl: the number of sampled points.
    Output:
        idx_spl: the indices of sampled points after the furthest point sampling. Shape: (batch_size, num_spl).
    '''
    return pcops.furthest_point_sample(pts_all, num_spl).long()
    # b_s, n_p, d_p, n_s, device = *pts_all.size(), num_spl, pts_all.device
    # idx_spl = torch.zeros(b_s, n_s, dtype=torch.long, device=device)
    # distance = torch.full((b_s, n_p), 1e10, device=device)
    # # furthest = torch.randint(0, n_p, (b_s, ), dtype=torch.long, device=device)
    # furthest = torch.zeros((b_s, ), dtype=torch.long, device=device)
    # batch_idx = torch.arange(b_s, dtype=torch.long, device=device)
    # for i in range(n_s):
    #     idx_spl[:, i] = furthest
    #     centroid = pts_all[batch_idx, furthest, :].reshape(b_s, 1, d_p)
    #     dist = torch.sum((pts_all - centroid) ** 2, -1)
    #     distance = torch.min(distance, dist)
    #     furthest = torch.max(distance, -1)[1]
    # return idx_spl


def square_distance(pts_1: Tensor, pts_2: Tensor) -> Tensor:
    '''
    Description:
        Calculate the square distances between points_1 and points_2.
    Input:
        pts_1: the values (features or coordinates) of points_1. Shape: (batch_size, num_pts_1, dim_pts).
        pts_2: the values (features or coordinates) of points_2. Shape: (batch_size, num_pts_2, dim_pts).
    Output:
        dist: the square distances between points_1 and points_2. Shape: (batch_size, num_pts_1, num_pts_2).
    '''
    b_s, n_p_1, n_p_2 = pts_1.size(0), pts_1.size(1), pts_2.size(1)
    dist = -2 * torch.matmul(pts_1, pts_2.permute(0, 2, 1))
    dist += torch.sum(pts_1 ** 2, -1).view(b_s, n_p_1, 1)
    dist += torch.sum(pts_2 ** 2, -1).view(b_s, 1, n_p_2)
    return dist


def knn_query(pts_all: Tensor, pts_qry: Tensor, num_grp: int) -> Tensor:
    '''
    Description:
        Calculate the `num_grp` neighbors of each point with knn query.
    Input:
        pts_all: the values (features or coordinates) of all points. Shape: (batch_size, num_all, dim_pts).
        pts_qry: the values (features or coordinates) of query points. Shape: (batch_size, num_qry, dim_pts).
        num_grp: the number of neighbors.
    Output:
        idx_grp: the indices of neighbors of each point after knn query. Shape: (batch_size, num_qry, num_grp).
    '''
    idx_grp = torch.topk(square_distance(pts_qry, pts_all), num_grp, dim=-1, largest=False, sorted=False)[1]
    return idx_grp


def ball_query(pts_all: Tensor, pts_qry: Tensor, num_grp: int, radius: float) -> Tensor:
    '''
    Description:
        Calculate the `num_grp` neighbors of each point with ball query.
    Input:
        pts_all: the values (features or coordinates) of all points. Shape: (batch_size, num_all, dim_pts).
        pts_qry: the values (features or coordinates) of query points. Shape: (batch_size, num_qry, dim_pts).
        num_grp: the number of neighbors.
    Output:
        idx_grp: the indices of neighbors of each point after ball query. Shape: (batch_size, num_qry, num_grp).
    '''
    idx_grp = pcops.ball_query(radius, num_grp, pts_all, pts_qry).long()
    # return idx_grp
    b_s, n_p, n_q, n_g, device = pts_all.size(0), pts_all.size(1), pts_qry.size(1), num_grp, pts_all.device
    idx_grp = torch.arange(n_p, dtype=torch.long, device=device).reshape(1, 1, n_p).repeat([b_s, n_q, 1])
    idx_grp[square_distance(pts_qry, pts_all) > radius * radius] = n_p
    idx_grp = idx_grp.sort(dim=-1)[0][:, :, :n_g]
    mask = (idx_grp == n_p)
    idx_grp[mask] = idx_grp[:, :, 0].view(b_s, n_q, 1).repeat([1, 1, n_g])[mask]
    return idx_grp


def gather_points(pts: Tensor, idx: Tensor) -> Tensor:
    '''
    Description:
        Gather points from index.
    Input:
        pts: the values (features or coordinates) of points. Shape: (batch_size, num_pts, dim_pts).
        idx: the indices of gathered points. Shape: (batch_size, num_qry, [num_grp]).
    Output:
        res: the values (features or coordinates) of gathered points. Shape: (batch_size, num_qry, [num_grp], dim_pts).
    '''
    sz = idx.size()
    res = torch.gather(pts, 1, idx.reshape(sz[0], -1, 1).expand(-1, -1, pts.size(-1))).reshape(*sz, -1)
    return res


def three_interpolation(c_hr: Tensor, c_lr: Tensor, f_lr: Tensor) -> Tensor:
    '''
    Description:
        Calculate the features of high resolution points with three interpolation.
    Input:
        c_lr (coordinate_high_resolution): the coordinates of high resolution points. Shape: (batch_size, num_high_res, dim_coord).
        c_lr (coordinate_low_resolution): the coordinates of low resolution points. Shape: (batch_size, num_low_res, dim_coord).
        f_lr (feature_low_resolution): the features of low resolution points. Shape: (batch_size, num_low_res, dim_feat).
    Output:
        f_hr (feature_high_resolution): the features of high resolution points. Shape: (batch_size, num_low_res, dim_feat).
    '''
    dist, idx = pcops.three_nn(c_hr, c_lr)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
    f_hr = torch.sum(gather_points(f_lr, idx.long()) * weight.view(c_hr.size(0), c_hr.size(1), 3, 1), dim=2)
    return f_hr


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # FOR PointCloud Operations
    print(furthest_point_sampling(torch.rand(8, 64, 3).cuda(), 16).shape)
    print(square_distance(torch.rand(8, 64, 3).cuda(), torch.rand(8, 64, 3).cuda()).shape)
    print(knn_query(torch.rand(8, 64, 3).cuda(), torch.rand(8, 16, 3).cuda(), 4).shape)
    print(ball_query(torch.rand(8, 64, 3).cuda(), torch.rand(8, 16, 3).cuda(), 4, 0.1).shape)
    print(gather_points(torch.rand(8, 64, 3).cuda(), torch.randint(0, 64, (8, 16), dtype=torch.int64).cuda()).shape)
    print(gather_points(torch.rand(8, 64, 3).cuda(), torch.randint(0, 64, (8, 16, 64), dtype=torch.int64).cuda()).shape)
    print(three_interpolation(torch.rand(8, 64, 3).cuda(), torch.rand(8, 16, 3).cuda(), torch.rand(8, 16, 16).cuda()).shape)