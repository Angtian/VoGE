from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from VoGE import _C
import math


def ray_tracing(transforms, points, isigmas, rays, image_size, thr: float, n_assign: int,
                bin_size: Optional[int] = None, max_points_per_bin: Optional[int] = None):
    if bin_size is None:
        max_image_size = max(image_size)
        bin_size = max(int(2 ** np.ceil(np.log2(max_image_size) - 5)), 10)
        
    if max_points_per_bin is None:
        max_points_per_bin = min(int(max(n_assign * 10, (points.shape[0]) / 10)), points.shape[0])

    # Without coarse stage
    if max_points_per_bin == -1:
        point_idx_size = ((image_size[0] - 1) // bin_size + 1, (image_size[1] - 1) // bin_size + 1)
        pixels_to_points_idx = torch.arange(points.shape[0]).view(1, 1, 1, -1) \
                                .expand(-1, point_idx_size[0], point_idx_size[1],-1) \
                                .type(torch.int32).contiguous().to(points.device)
    else:
        pixels_to_points_idx = rasterize_coarse(transforms, points, isigmas, image_size, thr, bin_size, max_points_per_bin)

    return ray_tracing_fine(points, isigmas, rays, pixels_to_points_idx, thr, bin_size, n_assign)


def convert_to_box(isigmas, thr, z, matrix):
    # [n, 2, 2]
    get = -np.log(thr) * matrix[:, :2, :2] @ torch.inverse(isigmas[:, :2, :2]) @ matrix[:, :2, :2]

    # [n, 1, 2] @ [n, 2, 2] -> [n, 2]
    boxes = (torch.ones((isigmas.shape[0], 1, 2), device=isigmas.device) @ get).pow(.5).squeeze(1) * z.unsqueeze(1)
    return boxes


def rasterize_coarse(cameras, points, isigmas, image_size, thr, bin_size, max_points_per_bin):
    n = points.shape[0]

    # Perspective cameras are default to be in ndc, however this will cause to_ndc_transform be inactivated
    _in_ndc_ori = cameras._in_ndc
    cameras._in_ndc = False

    to_ndc_transform = cameras.get_ndc_camera_transform()
    transforms = cameras.get_projection_transform().compose(to_ndc_transform)
    points_ndc = -transforms.transform_points(points)

    boxes = convert_to_box(isigmas, thr, -points_ndc[:, -1], transforms.get_matrix())

    points_ndc[..., 2] = points[..., 2]

    cloud_to_point = torch.zeros(1, dtype=torch.long).to(points.device)
    num_points_per_cloud = torch.ones(1, dtype=torch.long).to(points.device) * points.shape[0]

    pixels_to_points_idx = _RasterizeCoarse.apply(
        points_ndc,
        cloud_to_point,
        num_points_per_cloud,
        image_size,
        boxes,
        bin_size,
        max_points_per_bin
    )

    cameras._in_ndc = _in_ndc_ori

    return pixels_to_points_idx


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


def ray_tracing_fine(mus, isigmas, rays, bin_points, thr, bin_size, n_assign, inf=1e10):
    if mus.dim() == 3:
        assert isigmas.dim() == 4
        assert mus.shape[0] == 1
        mus = mus.squeeze(0)
        isigmas = isigmas.squeeze(0)

    if rays.dim() == 4:
        assert rays.shape[0] == 1
        rays = rays.squeeze(0)

    if bin_points.dim() == 4:
        assert bin_points.shape[0] == 1
        bin_points = bin_points.squeeze(0)

    assert mus.shape[0] == isigmas.shape[0]

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


class _RayTraceVoGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                mus,  # (P, 3)
                isigmas,  # (P, 3, 3)
                rays,  # (H, W, 3)
                bin_points,  # (BH, BW, M)
                thr_act,
                bin_size,
                n_assign
        ):
        args = (
            mus,  # (P, 3)
            isigmas,  # (P, 3, 3)
            rays,  # (H, W, 3)
            bin_points,  # (BH, BW, M)
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
