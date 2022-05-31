import torch
import numpy as np
from VoGE.Converter import Converters, Cuboid
from VoGE.Renderer import GaussianRenderSettings, GaussianRenderer, interpolate_attr, to_white_background
from VoGE.Meshes import GaussianMeshes, GaussianMeshesNaive
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
import matplotlib.pyplot as plt
import torch
import imageio
from PIL import Image
import os


def save_image(get, save_name):
    get = get.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255
    Image.fromarray(get.astype(np.uint8)).save(save_name)


if __name__ == '__main__':
    device = 'cuda'

    # Increase percentage will reduce transparency, however, transparency is not necessary to make this experiment works
    # We have test percentage = 0.85 (barely transparent), it still work, just need a little more iteration to converge
    percentage = 0.7

    colors_ = np.array([[0, 0.2, 1], [0, 0.2, 1], [0, 1, 0.2], [0, 1, 0.2], [0, 1, 1], [0, 1, 1]])
    verts, sigmas, colors = Cuboid.cuboid_gauss((-0.8, 0.8), (-0.4, 0.4),(-0.6, 0.6), 4000, colors=colors_, percentage=percentage)

    verts = torch.from_numpy(verts).type(torch.float32).to(device)
    sigmas = torch.from_numpy(sigmas).type(torch.float32).to(device)
    colors = torch.from_numpy(colors).type(torch.float32).to(device)

    colors_ = np.array([[1, 0.2, 0], [1, 0.2, 0], [1, 1, 0], [1, 1, 0], [0.2, 1, 0], [0.2, 1, 0]])
    verts1, sigmas1, colors1 = Cuboid.cuboid_gauss((-1, 1), (-1, 1), (-0.3, 0.3), 3000, colors=colors_, percentage=percentage)

    verts1 = torch.from_numpy(verts1).type(torch.float32).to(device)
    sigmas1 = torch.from_numpy(sigmas1).type(torch.float32).to(device)
    colors1 = torch.from_numpy(colors1).type(torch.float32).to(device)

    v_base0 = torch.Tensor([[0.5, 0, 1]]).to(device)
    v_base1 = torch.Tensor([[0, 0, 0]]).to(device)

    R, T = look_at_view_transform(dist=5, elev=10, azim=20, device=device)
    R1, T1 = look_at_view_transform(dist=6, elev=10, azim=-50, device=device)
    T1 += torch.Tensor([[-1, 0, 0]]).to(device)

    save_head = 'reason_occ/VoGE'
    save_pendix = '_1'

    os.makedirs('reason_occ', exist_ok=True)

    image_size = (400, 400)
    camera = PerspectiveCameras(focal_length=300, in_ndc=False, device=device, image_size=(image_size, ), principal_point=((image_size[0] // 2, image_size[1] // 2), ), T=T, R=R)
    render_settings = GaussianRenderSettings(max_assign=60, principal=(image_size[0] // 2, image_size[1] // 2), image_size=image_size, max_point_per_bin=1500)
    renderer = GaussianRenderer(cameras=camera, render_settings=render_settings)

    gmesh = GaussianMeshesNaive(verts=torch.cat((verts + v_base0, verts1 + v_base1), dim=0), sigmas=torch.cat((sigmas, sigmas1), dim=0))
    frag = renderer(gmesh)

    timg = interpolate_attr(frag, torch.cat((colors, colors1), dim=0)).detach()

    with torch.no_grad():
        verts_ = torch.cat((verts + v_base0, verts1 + v_base1), dim=0)
        sigmas_ = torch.cat((sigmas, sigmas1), dim=0)
        gmesh = GaussianMeshesNaive(verts=verts_, sigmas=sigmas_)
        frag = renderer(gmesh, R=R, T=T)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_target_V0' + save_pendix + '.png')

        frag = renderer(gmesh, R=R1, T=T1)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_target_V1' + save_pendix + '.png')

    # v_pred0 = torch.Tensor([[-1, 0, -10]]).to(device) also works
    v_pred0 = torch.Tensor([[-1, 0, -5]]).to(device)
    v_pred1 = torch.Tensor([[0, 0, 0]]).to(device)

    v_pred1.requires_grad = True
    v_pred0.requires_grad = True

    optimizer = torch.optim.Adam([v_pred0, v_pred1], lr=0.05, betas=(0.6, 0.4))

    l2loss = torch.nn.MSELoss()

    with torch.no_grad():
        verts_ = torch.cat((verts + v_pred0, verts1 + v_pred1), dim=0)
        sigmas_ = torch.cat((sigmas, sigmas1), dim=0)
        gmesh = GaussianMeshesNaive(verts=verts_, sigmas=sigmas_)
        frag = renderer(gmesh, R=R, T=T)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_before_V0' + save_pendix + '.png')

        save_image(timg, save_head + '_target' + save_pendix + '.png')

        frag = renderer(gmesh, R=R1, T=T1)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_before_V1' + save_pendix + '.png')

    out_image = []

    for iter_ in range(200):
        verts_ = torch.cat((verts + v_pred0, verts1 + v_pred1), dim=0)
        sigmas_ = torch.cat((sigmas, sigmas1), dim=0)
        gmesh = GaussianMeshesNaive(verts=verts_, sigmas=sigmas_)
        frag = renderer(gmesh, R=R, T=T)
        img = interpolate_attr(frag, torch.cat((colors, colors1), dim=0))

        loss_ = l2loss(img, timg)
        loss_.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(v_pred0)
        im_np = torch.cat((img, timg), dim=1).clamp(min=0, max=1).detach().cpu().numpy() * 255
        out_image.append(im_np.astype(np.uint8))

    imageio.mimwrite(save_head + '_2cuboid%s.mp4' % save_pendix, out_image, fps=24, quality=8)

    with torch.no_grad():
        verts_ = torch.cat((verts + v_pred0, verts1 + v_pred1), dim=0)
        sigmas_ = torch.cat((sigmas, sigmas1), dim=0)
        gmesh = GaussianMeshesNaive(verts=verts_, sigmas=sigmas_)
        frag = renderer(gmesh, R=R, T=T)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_after_V0' + save_pendix + '.png')

        img = interpolate_attr(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_optimized' + save_pendix + '.png')

        frag = renderer(gmesh, R=R1, T=T1)
        img = to_white_background(frag, torch.cat((colors, colors1), dim=0))
        save_image(img, save_head + '_after_V1' + save_pendix + '.png')

