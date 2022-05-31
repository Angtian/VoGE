import torch
import numpy as np
from VoGE.Converter import Converters
from VoGE.Converter import Cuboid
from VoGE.Renderer import GaussianRenderSettings, GaussianRenderer, interpolate_attr, to_white_background
from VoGE.Meshes import GaussianMeshes, GaussianMeshesNaive
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
import matplotlib.pyplot as plt
import torch
import imageio
from PIL import Image
import random
import tqdm
import os


def to_sym(matirx_):
    return torch.tril(matirx_) @ torch.tril(matirx_).transpose(-2, -1)


def get_template():
    x = torch.Tensor([0  , 0.4, 0.6, 0.85])
    y = torch.Tensor([0.85, 0.6, 0.4, 0.85])

    out0 = torch.cat([torch.zeros(1), x, -x, y, -y])
    out1 = torch.cat([torch.zeros(1), y, -y, -x, x])

    return out0, out1


def efficient_cuboid(scale=1):
    tem0, tem1 = get_template()

    get = []
    get.append(torch.stack([tem0, tem1, -torch.ones_like(tem0)]).T)
    get.append(torch.stack([tem0, tem1, torch.ones_like(tem0)]).T)
    get.append(torch.stack([tem0, -torch.ones_like(tem0), tem1]).T)
    get.append(torch.stack([tem0, torch.ones_like(tem0), tem1]).T)
    get.append(torch.stack([-torch.ones_like(tem0), tem0, tem1]).T)
    get.append(torch.stack([torch.ones_like(tem0), tem0, tem1]).T)

    return torch.cat(get, dim=0) * scale, tem0.shape[0]


if __name__ == '__main__':
    device = 'cuda'
    
    os.makedirs('image_fitting', exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    image_size = (256, 256)

    rgb_mapping = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.8, 0.8], [0.8, 0, 0.8], [0.8, 0.8, 0]]).type(torch.float32).to(device)
    colors_0 = np.eye(6, dtype=np.float32)
    verts, sigmas, colors = Cuboid.cuboid_gauss((-1, 1), (-1, 1),(-1, 1), 4000, colors=colors_0, percentage=0.7)

    verts_ = torch.from_numpy(verts).type(torch.float32).to(device)
    sigmas_ = torch.from_numpy(sigmas).type(torch.float32).to(device)
    colors_ = torch.from_numpy(colors).type(torch.float32).to(device)

    t_mesh = GaussianMeshesNaive(verts=verts_, sigmas=sigmas_)

    verts, kn = efficient_cuboid()

    sig_init = torch.eye(3)[None].expand(verts.shape[0], -1, -1) * 4
    for i in range(6):
        sig_init[i * kn] /= 3
    verts = verts.to(device)

    sig_ori = torch.nn.Parameter(sig_init.to(device).clone(), requires_grad=True)


    camera = PerspectiveCameras(focal_length=200, in_ndc=False, device=device, image_size=(image_size, ), principal_point=((image_size[0] // 2, image_size[1] // 2), ))

    render_settings = GaussianRenderSettings(max_assign=50, principal=(image_size[0] // 2, image_size[1] // 2), image_size=image_size, max_point_per_bin=1500)
    renderer = GaussianRenderer(cameras=camera, render_settings=render_settings)

    render_settings1 = GaussianRenderSettings(max_assign=verts.shape[0], principal=(image_size[0] // 2, image_size[1] // 2), image_size=image_size, max_point_per_bin=-1, thr_activation=0)
    renderer1 = GaussianRenderer(cameras=camera, render_settings=render_settings1)

    idx_ = torch.from_numpy(colors_0).to(device).unsqueeze(1).expand(-1, kn, -1).contiguous().view(-1, 6)

    optimizer = torch.optim.Adam([sig_ori], lr=0.02, betas=(0.8, 0.6))
    criteria = torch.nn.L1Loss()

    out_images = []

    rand_para = [[-90, 0], [0, 0], [90, 0], [0, 90], [0, 180], [0, 270]]

    pbar = tqdm.tqdm(range(3200))
    for i in pbar:
        # R, T = look_at_view_transform(5, random.randrange(-90, 90), random.randrange(0, 360), device=device)

        if i <= 1500:
            idx_rand = random.randint(0, 5)
            R, T = look_at_view_transform(5, rand_para[idx_rand][0], rand_para[idx_rand][1], device=device)
        else:
            R, T = look_at_view_transform(5, random.randrange(-60, 60), random.randrange(0, 360), device=device)

        with torch.no_grad():
            frag = renderer(t_mesh, R=R, T=T)
            t_idx_map = interpolate_attr(frag, colors_)

        sigmas = to_sym(sig_ori)

        gmesh = GaussianMeshesNaive(verts, sigmas)

        frag = renderer1(gmesh, R=R, T=T)
        g_idx_map = interpolate_attr(frag, idx_)

        loss = criteria(g_idx_map, t_idx_map)
        loss.backward()

        # print(loss.item())

        if (i + 1) % 10 == 0:
            pbar.set_description('loss:%.4f' % loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                R, T = look_at_view_transform(4, 20, 30, device=device)

                gmesh = GaussianMeshesNaive(verts, sigmas)

                frag = renderer1(gmesh, R=R, T=T)
                g_idx_map = interpolate_attr(frag, idx_)

                img = torch.einsum("hwk, kc->hwc", g_idx_map, rgb_mapping)

                get = img.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255

                out_images.append(get.astype(np.uint8))

    imageio.mimwrite('image_fitting/cuboid_fitting.mp4', out_images, fps=24, quality=8)

    out_images = []
    with torch.no_grad():
        for i in range(180):
            sigmas = to_sym(sig_ori)
            R, T = look_at_view_transform(4, 10, i * 2, device=device)

            gmesh = GaussianMeshesNaive(verts, sigmas)

            frag = renderer1(gmesh, R=R, T=T)
            g_idx_map = interpolate_attr(frag, idx_)

            img = torch.einsum("hwk, kc->hwc", g_idx_map, rgb_mapping)

            get = img.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255

            out_images.append(get.astype(np.uint8))

        for i in range(40):
            sigmas = to_sym(sig_ori)
            R, T = look_at_view_transform(4, i + 10, 0, device=device)

            gmesh = GaussianMeshesNaive(verts, sigmas)

            frag = renderer1(gmesh, R=R, T=T)
            g_idx_map = interpolate_attr(frag, idx_)

            img = torch.einsum("hwk, kc->hwc", g_idx_map, rgb_mapping)

            get = img.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255

            out_images.append(get.astype(np.uint8))

        for i in range(30):
            sigmas = to_sym(sig_ori)
            R, T = look_at_view_transform(4, (50-i), 0, device=device)

            gmesh = GaussianMeshesNaive(verts, sigmas)

            frag = renderer1(gmesh, R=R, T=T)
            g_idx_map = interpolate_attr(frag, idx_)

            img = torch.einsum("hwk, kc->hwc", g_idx_map, rgb_mapping)

            get = img.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255

            out_images.append(get.astype(np.uint8))

        for i in range(idx_.shape[0]):
            mask1 = torch.Tensor([0] * i + [1] * (idx_.shape[0] - i)).view(-1, 1).to(device)
            # print(mask1)
            sigmas = to_sym(sig_ori)
            R, T = look_at_view_transform(4, 20, i, device=device)

            gmesh = GaussianMeshesNaive(verts, sigmas)

            frag = renderer1(gmesh, R=R, T=T)
            g_idx_map = interpolate_attr(frag, idx_ * mask1)

            img = torch.einsum("hwk, kc->hwc", g_idx_map, rgb_mapping)

            get = img.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255

            out_images.append(get.astype(np.uint8))
            out_images.append(get.astype(np.uint8))

    imageio.mimwrite('image_fitting/cuboid_result.mp4', out_images, fps=24, quality=8)



