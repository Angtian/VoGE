from VoGE.Aggregation import *
from VoGE.RayTracing import rasterize_coarse, ray_tracing
from VoGE.Utils import Batchifier, DataParallelBatchifier
import torch
import torch.nn as nn
from typing import NamedTuple, Optional, Tuple, Union, List
from pytorch3d.renderer.implicit.raysampling import NDCMultinomialRaysampler


# from pytorch3d.renderer.cameras import CamerasBase


class Fragments(object):
    vert_weight: torch.Tensor  # [k, M]
    vert_index: torch.Tensor  # [k, M]
    valid_num: torch.Tensor  # [k]

    def __init__(self, vert_weight, vert_index, valid_num, vert_hit_length):
        self.vert_weight = vert_weight
        self.vert_index = vert_index
        self.valid_num = valid_num
        self.vert_hit_length = vert_hit_length

    def __getitem__(self, item):
        assert len(self.valid_num.shape) == 3, 'Index access is only available when batched.'
        return Fragments(vert_weight=self.vert_weight[item], vert_index=self.vert_index[item], valid_num=self.valid_num[item], vert_hit_length=self.vert_hit_length[item])

    def __len__(self):
        return self.valid_num.shape[0]

    @property
    def shape(self):
        return (self.vert_weight.shape, self.vert_index.shape, self.valid_num.shape, self.vert_hit_length.shape)

    def squeeze(self):
        assert self.valid_num.shape[0] == 1
        return self.__getitem__(0)

    def unsqueeze(self):
        assert len(self.valid_num.shape) == 2
        return Fragments(vert_weight=self.vert_weight.unsqueeze(0), vert_index=self.vert_index.unsqueeze(0),
                         valid_num=self.valid_num.unsqueeze(0), vert_hit_length=self.vert_hit_length.unsqueeze(0))

    def to_dict(self):
        return dict(vert_weight=self.vert_weight, vert_index=self.vert_index,
                         valid_num=self.valid_num, vert_hit_length=self.vert_hit_length)

    def copy(self):
        return Fragments(vert_weight=self.vert_weight.contiguous(), vert_index=self.vert_index.contiguous(),
                         valid_num=self.valid_num.contiguous(), vert_hit_length=self.vert_hit_length.contiguous())


class GaussianRenderSettings:
    __slots__ = ['image_size',
                 'max_assign',
                 'thr_activation',
                 'absorptivity',
                 'inverse_sigma',
                 'principal',
                 'max_point_per_bin']

    def __init__(self,
                 image_size: Union[int, Tuple[int, int]] = 256,
                 max_assign: int = 20,
                 thr_activation: float = 0.01,
                 absorptivity: float = 1,
                 inverse_sigma: bool = False,
                 principal: Union[None, Tuple[int, int], Tuple[float, float]] = None,
                 max_point_per_bin: Union[None, int] = None,
                 **kwargs):
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size
        self.max_assign = max_assign
        self.thr_activation = thr_activation
        self.absorptivity = absorptivity
        self.inverse_sigma = inverse_sigma
        self.principal = principal
        self.max_point_per_bin = max_point_per_bin

    def __getitem__(self, item):
        return self.__getattribute__(item)


class GaussianRenderer(nn.Module):
    to_set_args = ['R', 'T', 'focal', 'principal']

    def __init__(self, cameras, render_settings: Union[dict, GaussianRenderSettings]):
        super(GaussianRenderer, self).__init__()
        self.cameras = cameras
        self.render_settings = render_settings
        self.device = cameras.device

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        self.cameras = self.cameras.to(device)
        self.device = device
        return self

    def forward(self, gmeshes, **kwargs):
        assert not self.cameras.in_ndc(), 'Got NDC camera. Cameras.in_ndc must be set to false.'
        for k_arg in kwargs.keys():
            if k_arg in self.to_set_args:
                if isinstance(kwargs[k_arg], torch.Tensor):
                    setattr(self.cameras, k_arg, kwargs[k_arg].to(self.device))
                else:
                    setattr(self.cameras, k_arg, kwargs[k_arg])

        verts, sigmas, radians = gmeshes()

        map_size = self.render_settings['image_size']

        sigmas = expend_sigma(sigmas)
        if self.render_settings['principal'] is None:
            principal = (self.cameras.principal_point[0][1].item(), self.cameras.principal_point[0][0].item())
        else:
            principal = self.render_settings['principal']
        
        if len(verts.shape) == 2:
            verts = verts[None]
        
        ray_sampler = NDCMultinomialRaysampler(image_width=int(self.render_settings['image_size'][1]), image_height=int(self.render_settings['image_size'][0]), unit_directions=True, n_pts_per_ray=1, min_depth=0, max_depth=10)
        
        ray_bundle = ray_sampler(self.cameras, )
        rays = ray_bundle.directions # (B, H, W, 3)
        ray_origins = ray_bundle.origins[:, 0, 0, :] # (B, 3)

        verts_transformed = verts - ray_origins[:, None]
        if len(sigmas.shape) == 3:
            sigmas = sigmas.unsqueeze(0).expand(verts_transformed.shape[0], -1, -1, -1)

        if self.render_settings['inverse_sigma']:
            isigma = 2 * torch.inverse(sigmas)
        else:
            isigma = 2 * sigmas

        # [b, h, w, s]
        sel_idx, sel_len, sel_act, sel_dsd = ray_tracing(
                self.cameras, verts_transformed, isigma, rays,
                map_size, thr=self.render_settings['thr_activation'], n_assign=self.render_settings['max_assign'],
                max_points_per_bin=self.render_settings['max_point_per_bin'])
        
        sel_idx = sel_idx.clone()

        vert_weight, vert_index, valid_num, vert_hit_length = aggregation(sel_idx=sel_idx, sel_act=sel_act, sel_len=sel_len, 
                                                                        sel_dsd=sel_dsd, occupation_weight=self.render_settings['absorptivity'])

        return Fragments(vert_weight=vert_weight, vert_index=vert_index, valid_num=valid_num, vert_hit_length=vert_hit_length)


def interpolate_attr(fragments: Fragments, vert_attr: torch.Tensor):
    return merge_final(vert_attr=vert_attr, weight=fragments.vert_weight, valid_num=fragments.valid_num, vert_assign=fragments.vert_index)


def get_silhouette(fragments: Fragments):
    merged_weight = fragments.vert_weight.sum(-1)
    return torch.min(merged_weight, torch.ones_like(merged_weight))


def to_colored_background(fragments: Fragments, colors: torch.Tensor, background_color: Union[torch.Tensor, tuple, list] = (1, 1, 1), thr: float = -1):
    masks = get_silhouette(fragments).unsqueeze(-1)
    if not torch.is_tensor(background_color):
        background_color = torch.Tensor(list(background_color) if isinstance(background_color, tuple) else background_color).to(colors.device)

    if thr > 0:
        masks = (masks > thr).type_as(masks)

    rgb = interpolate_attr(fragments, colors)
    return torch.min(rgb + torch.ones_like(rgb) * (1 - masks) * background_color, torch.ones_like(rgb))


def to_white_background(fragments: Fragments, colors: torch.Tensor, thr: float = -1):
    background_color = (1, 1, 1)
    return to_colored_background(fragments=fragments, colors=colors, background_color=background_color, thr=thr)
