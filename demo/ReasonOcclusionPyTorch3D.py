import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
import matplotlib.pyplot as plt
import imageio
from pytorch3d.renderer.blending import softmax_rgb_blend, hard_rgb_blend, BlendParams
from PIL import Image
from VoGE.Converter import Cuboid
import os


def interpolate_attributes_as_img(fragments, face_colors, **kwargs):
    colors_img = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_colors)
    blend_params = BlendParams(**kwargs)
    if colors_img.shape[3] == 1:
        return hard_rgb_blend(colors_img, fragments, blend_params).squeeze()
    else:
        return softmax_rgb_blend(colors_img, fragments, blend_params).squeeze()


def save_image(get, save_name):
    get = get.clamp(min=0, max=1)[..., :3].detach().cpu().numpy() * 255
    Image.fromarray(get.astype(np.uint8)).save(save_name)


if __name__ == '__main__':
    device = 'cuda'

    colors_ = np.array([[0, 0.2, 1], [0, 0.2, 1], [0, 1, 0.2], [0, 1, 0.2], [0, 1, 1], [0, 1, 1]])
    verts, faces, colors = Cuboid.cuboid_mesh((-0.8, 0.8), (-0.4, 0.4),(-0.6, 0.6), 4000, colors=colors_)

    verts = torch.from_numpy(verts).type(torch.float32).to(device)
    faces = torch.from_numpy(faces).type(torch.long).to(device)
    colors = torch.from_numpy(colors).type(torch.float32).to(device)

    faces_color0 = colors[faces]

    colors_ = np.array([[1, 0.2, 0], [1, 0.2, 0], [1, 1, 0], [1, 1, 0], [0.2, 1, 0], [0.2, 1, 0]])
    verts1, faces1, colors1 = Cuboid.cuboid_mesh((-1, 1), (-1, 1), (-0.3, 0.3), 3000, colors=colors_)

    n_v0, n_v1 = verts.shape[0], verts1.shape[0]

    verts1 = torch.from_numpy(verts1).type(torch.float32).to(device)
    faces1 = torch.from_numpy(faces1).type(torch.long).to(device)
    colors1 = torch.from_numpy(colors1).type(torch.float32).to(device)

    v_base0 = torch.Tensor([[0.5, 0, 1]]).to(device)
    v_base1 = torch.Tensor([[0, 0, 0]]).to(device)

    faces_color1 = colors1[faces1]

    faces_colors = torch.cat((faces_color0, faces_color1), dim=0)

    R, T = look_at_view_transform(dist=5, elev=10, azim=20, device=device)
    R1, T1 = look_at_view_transform(dist=6, elev=10, azim=-50, device=device)

    T1 += torch.Tensor([[-1, 0, 0]]).to(device)

    image_size = (400, 400)

    save_head = 'reason_occ/SoftRas'
    save_pendix = '_1'

    os.makedirs('reason_occ', exist_ok=True)

    # The default gamma is 1e-4, however, which make it hardly update.
    # we have tried 1e-2, which makes it converge to some other local min
    faces_per_pixel = 60
    sigmas = 1e-3
    gamma = 1e-3

    camera = PerspectiveCameras(focal_length=300, in_ndc=False, device=device, image_size=(image_size, ), principal_point=((image_size[0] // 2, image_size[1] // 2), ), T=T, R=R)
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel, blur_radius=sigmas)
    raster = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

    meshes = Meshes(verts=[verts, verts1], faces=[faces, faces1])
    meshes = Meshes(verts=[meshes.verts_packed()], faces=[meshes.faces_packed()])

    off_set = torch.cat((v_base0.expand(n_v0, -1), v_base1.expand(n_v1, -1)), dim=0)

    new_meshes = meshes.offset_verts(off_set)

    frag = raster(new_meshes)

    timg = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(0., 0., 0.)).detach()

    v_pred0 = torch.Tensor([[-1, 0, -5]]).to(device)
    v_pred1 = torch.Tensor([[0, 0, 0]]).to(device)

    v_pred1.requires_grad = True
    v_pred0.requires_grad = True

    optimizer = torch.optim.Adam([v_pred0, v_pred1], lr=0.05, betas=(0.6, 0.4))

    l2loss = torch.nn.MSELoss()


    with torch.no_grad():
        off_set = torch.cat((v_pred0.expand(n_v0, -1), v_pred1.expand(n_v1, -1)), dim=0)
        new_meshes = meshes.offset_verts(off_set)
        frag = raster(new_meshes, R=R, T=T)
        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(1., 1., 1.))
        save_image(img, save_head + '_before_V0' + save_pendix + '.png')

        save_image(timg, save_head + '_target' + save_pendix + '.png')

        frag = raster(new_meshes, R=R1, T=T1)
        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(1., 1., 1.))
        save_image(img, save_head + '_before_V1' + save_pendix + '.png')

    out_image = []

    for iter_ in range(200):
        off_set = torch.cat((v_pred0.expand(n_v0, -1), v_pred1.expand(n_v1, -1)), dim=0)
        new_meshes = meshes.offset_verts(off_set)

        frag = raster(new_meshes, R=R, T=T)

        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(0., 0., 0.))

        loss_ = l2loss(img, timg)
        loss_.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(v_pred0)
        im_np = torch.cat((img, timg), dim=1).clamp(min=0, max=1).detach().cpu().numpy() * 255
        out_image.append(im_np.astype(np.uint8))

    imageio.mimwrite(save_head + '_2cuboid%s.mp4' % save_pendix, out_image, fps=24, quality=8)

    with torch.no_grad():
        off_set = torch.cat((v_pred0.expand(n_v0, -1), v_pred1.expand(n_v1, -1)), dim=0)
        new_meshes = meshes.offset_verts(off_set)
        frag = raster(new_meshes, R=R, T=T)
        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(1., 1., 1.))
        save_image(img, save_head + '_after_V0' + save_pendix + '.png')

        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(0., 0., 0.))
        save_image(img, save_head + '_optimized' + save_pendix + '.png')

        frag = raster(new_meshes, R=R1, T=T1)
        img = interpolate_attributes_as_img(frag, faces_colors, sigma=sigmas, gamma=gamma, background_color=(1., 1., 1.))
        save_image(img, save_head + '_after_V1' + save_pendix + '.png')

    # plt.imshow(timg.detach().cpu().numpy())
    # plt.show()




