from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from VoGE import _C
import math


inf = 1e8


def ray_tracing(transforms, points, isigmas, rays, image_size, thr: float, n_assign: int,
                bin_size: Optional[int] = None, max_points_per_bin: Optional[int] = None, **kwargs):
    if bin_size is None:
        max_image_size = max(image_size)
        bin_size = max(int(2 ** np.ceil(np.log2(max_image_size) - 5)), 10)
        
    if max_points_per_bin is None:
        max_points_per_bin = min(int(max(n_assign * 10, (points.shape[1]) / 10)), points.shape[1])

    # Without coarse stage
    if max_points_per_bin == -1:
        point_idx_size = ((image_size[0] - 1) // bin_size + 1, (image_size[1] - 1) // bin_size + 1)
        pixels_to_points_idx = (torch.arange(points.shape[1]).view(1, 1, 1, -1) + torch.arange(points.shape[0]).view(-1, 1, 1, 1) * points.shape[1])\
                                .expand(-1, point_idx_size[1], point_idx_size[1], -1) \
                                .type(torch.int32).contiguous().to(points.device)
    else:
        pixels_to_points_idx = rasterize_coarse(transforms, points, isigmas, image_size, thr, bin_size, max_points_per_bin, **kwargs)

    return ray_tracing_fine(points.view(-1, 3), isigmas.view(-1, 3, 3), rays, pixels_to_points_idx, thr, bin_size, n_assign)


def convert_to_box(isigmas, thr, z, matrix):
    # [b, n, 2, 2]
    get = -np.log(thr) * matrix[:, None, :2, :2] @ torch.inverse(isigmas[:, :, :2, :2]) @ matrix[:, None, :2, :2]

    # [b, n, 1, 2] @ [b, n, 2, 2] -> [b, n, 2]
    boxes = (torch.ones((*isigmas.shape[0:2], 1, 2), device=isigmas.device) @ get).pow(.5).squeeze(2) * z.unsqueeze(-1)
    return boxes


def rasterize_coarse(cameras, points, isigmas, image_size, thr, bin_size, max_points_per_bin, cloud_to_point=None, num_points_per_cloud=None):
    n = points.shape[0]

    C = -torch.matmul(torch.inverse(cameras.R.transpose(1, 2)), cameras.T[:, :, None])
    points = points + C.view(-1, 1, 3)
    
    to_ndc_transform = cameras.get_ndc_camera_transform()
    transforms = cameras.get_full_projection_transform().compose(to_ndc_transform)
    points_ndc = -transforms.transform_points(points)

    rotation_ = cameras.get_world_to_view_transform().get_matrix()[:, :3, :3].unsqueeze(1).expand(-1, isigmas.shape[1], -1, -1)
    isigmas = rotation_.transpose(2, 3) @ isigmas @ rotation_

    boxes = convert_to_box(isigmas, thr, -points_ndc[..., -1], cameras.get_projection_transform().compose(to_ndc_transform).get_matrix())

    points_ndc[..., 2] = cameras.get_world_to_view_transform().transform_points(points)[..., 2]

    if cloud_to_point is None:
        cloud_to_point = torch.arange(points_ndc.shape[0], dtype=torch.long).to(points.device) * points_ndc.shape[1]
    if num_points_per_cloud is None:
        num_points_per_cloud = torch.ones(points_ndc.shape[0], dtype=torch.long).to(points.device) * points_ndc.shape[1]

    pixels_to_points_idx = _RasterizeCoarse.apply(
        points_ndc.view(-1, 3),
        cloud_to_point,
        num_points_per_cloud,
        image_size,
        boxes.view(-1, 2),
        bin_size,
        max_points_per_bin
    )
    return pixels_to_points_idx


def ray_tracing_fine(mus, isigmas, rays, bin_points, thr, bin_size, n_assign, inf=1e10):
    
    assert isigmas.dim() == 3
    assert mus.dim() == 2
    assert rays.dim() == 4
    assert bin_points.dim() == 4

    assert mus.shape[0] == isigmas.shape[0] and mus.shape[1] == 3 and isigmas.shape[1] == 3 and isigmas.shape[2] == 3

    thr_act = -math.log(thr + 1 / inf)

    return _RayTraceVoGE.apply(
        mus,
        isigmas,
        rays,
        bin_points,
        thr_act,
        bin_size,
        n_assign
    )

def ray_trace_voge_ray(mus, sigmas, rays):
    if isinstance(sigmas, float) or isinstance(sigmas, int):
        sigmas = torch.eye(3, device=mus.device)[None].expand(mus.shape[0], -1, -1) * sigmas
    if sigmas.dim() == 1:
        sigmas = sigmas.view(-1, 1, 1) * torch.eye(3, device=sigmas.device)[None]

    assert mus.is_cuda and sigmas.is_cuda and rays.is_cuda
    assert mus.dim() == 2 and mus.shape[1] == 3
    assert rays.dim() == 2 and rays.shape[1] == 3
    assert sigmas.dim() == 3 and sigmas.shape[1] == 3 and sigmas.shape[2] == 3

    return _RayTraceVoGERay.apply(mus, sigmas, rays)


def find_nearest_k(hit_len_in, hit_act_in, hit_dsd_in, K, thr):
    assert hit_len_in.is_cuda and hit_act_in.is_cuda and hit_dsd_in.is_cuda

    thr_act = -math.log(thr + 1 / inf)
    return _FindNearestK.apply(hit_len_in, hit_act_in, hit_dsd_in, thr_act, K)


def find_farest_k(hit_len_in, hit_act_in, hit_dsd_in, K, thr):
    assert hit_len_in.is_cuda and hit_act_in.is_cuda and hit_dsd_in.is_cuda

    thr_act = -math.log(thr + 1 / inf)
    point_idx, hit_len, hit_act, hit_dsd = _FindNearestK.apply(-hit_len_in, hit_act_in, hit_dsd_in, thr_act, K)
    return point_idx, -hit_len, hit_act, hit_dsd


class _RasterizeCoarse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                points_ndc,
                cloud_to_point,
                num_points_per_cloud,
                image_size,
                boxes,
                bin_size,
                max_points_per_bin
        ):
        args = (
            points_ndc,
            cloud_to_point,
            num_points_per_cloud,
            image_size,
            boxes,
            bin_size,
            max_points_per_bin
        )
        pixels_to_points_idx = _C.rasterize_points_coarse(*args)
        ctx.mark_non_differentiable(pixels_to_points_idx)
        return pixels_to_points_idx

    def backward(ctx, grad_idx):
        return (None, ) * 7


class _RayTraceVoGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                mus,  # (P, 3)
                isigmas,  # (P, 3, 3)
                rays,  # (B, H, W, 3)
                bin_points,  # (B, BH, BW, M)
                thr_act,
                bin_size,
                n_assign
        ):
        args = (
            mus,  # (P, 3)
            isigmas,  # (P, 3, 3)
            rays,  # (B, H, W, 3)
            bin_points,  # (B, BH, BW, M)
            thr_act,
            bin_size,
            n_assign
        )
        sel_idx, sel_len, sel_act, sel_dsd = _C.ray_trace_voge_fine(*args)
        ctx.save_for_backward(mus, isigmas, rays, sel_idx)
        ctx.mark_non_differentiable(sel_idx)
        return sel_idx, sel_len, sel_act, sel_dsd

    @staticmethod
    def backward(ctx,
                 grad_sel_idx,
                 grad_sel_len,
                 grad_sel_act,
                 grad_sel_dsd
        ):
        mus, isigmas, rays, sel_idx = ctx.saved_tensors
        args = (
            mus,
            isigmas,
            rays,
            sel_idx,
            grad_sel_len,
            grad_sel_act,
            grad_sel_dsd
        )
        grad_rays, grad_mus, grad_isg = _C.ray_trace_voge_fine_backward(*args)
        grads = (
            grad_mus,
            grad_isg,
            grad_rays,
            None,
            None,
            None,
            None
        )
        return grads


class _RayTraceVoGERay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigmas, rays):
        hit_len, hit_act, hit_dsd = _C.ray_trace_voge_ray(mus, sigmas, rays)
        ctx.save_for_backward(mus, sigmas, rays)
        return hit_len, hit_act, hit_dsd

    @staticmethod
    def backward(ctx, grad_hit_len, grad_hit_act, grad_hit_dsd):
        mus, sigmas, rays = ctx.saved_tensors
        grad_ray, grad_mus, grad_sig = _C.ray_trace_voge_ray_backward(mus, sigmas, rays, grad_hit_len, grad_hit_act, grad_hit_dsd)
        return grad_mus, grad_sig, grad_ray


class _FindNearestK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hit_len_in, hit_act_in, hit_dsd_in, thr, K):
        point_idx, hit_len, hit_act, hit_dsd = _C.find_nearest_k(hit_len_in, hit_act_in, hit_dsd_in, thr, K)
        ctx.save_for_backward(point_idx, hit_len_in)
        return point_idx, hit_len, hit_act, hit_dsd

    @staticmethod
    def backward(ctx, grad_point_idx, grad_hit_len, grad_hit_act, grad_hit_dsd):
        point_idx, hit_len_in = ctx.saved_tensors
        grad_hit_len_in = torch.zeros_like(hit_len_in)  # (M, N)
        grad_hit_act_in = torch.zeros_like(hit_len_in)  # (M, N)
        grad_hit_dsd_in = torch.zeros_like(hit_len_in)  # (M, N)
        point_idx = point_idx.long()

        grad_hit_len_in = grad_hit_len_in.scatter(dim=1, index=point_idx, src=grad_hit_len)
        grad_hit_act_in = grad_hit_len_in.scatter(dim=1, index=point_idx, src=grad_hit_act)
        grad_hit_dsd_in = grad_hit_len_in.scatter(dim=1, index=point_idx, src=grad_hit_dsd)
        return grad_hit_len_in, grad_hit_act_in, grad_hit_dsd_in, None, None


    
