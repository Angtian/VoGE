from PIL import Image
import numpy as np
import tqdm
import os
import torch

from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, interpolate_attr, to_white_background
from VoGE.Utils import rotation_theta
from VoGE.Meshes import GaussianMeshesNaive
from VoGE.Converter.IO import load_goff, load_off, pre_process_pascal, to_torch
from VoGE.Converter.Converters import naive_vertices_converter
from VoGE.Sampler import sample_features

from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
import numpy as np
from PIL import Image

device = 'cuda'
image_path = 'data/car_image.JPEG'
annos_path = 'data/car_annotation.npz'
cad_path = 'data/car.off'

annos = np.load(annos_path)
im = torch.from_numpy(np.array(Image.open(image_path)))

render_settings = GaussianRenderSettings(batch_size=-1, image_size=(256, 672), max_assign=80, )

cameras = PerspectiveCameras(focal_length=1800.0, principal_point=((336, 128),),
                             image_size=(render_settings['image_size'],), device=device, )

render = GaussianRenderer(cameras=cameras, render_settings=render_settings)

theta = float(annos['theta'])
azim = float(annos['azimuth'])
elev = float(annos['elevation'])
dist = 3

meshes = GaussianMeshesNaive(
    *to_torch(*naive_vertices_converter(*pre_process_pascal(*load_off(cad_path)), percentage=0.5, max_sig_rate=2)))
meshes = meshes.to(device)

R, T = look_at_view_transform([dist], [elev], [azim], degrees=False)
R = torch.bmm(R, rotation_theta(torch.Tensor([theta, ] * R.shape[0])))
with torch.no_grad():
    frag = render(meshes, R=R, T=T)

get, get_sum = sample_features(frag, im.type(torch.float32).to(device), meshes.verts.shape[0])
texture = get / get_sum[:, None] / 255
texture = texture * 0.7

print('Finished_texture!')

R, T = look_at_view_transform([dist], [elev], [azim - np.pi / 6], degrees=False)
R = torch.bmm(R, rotation_theta(torch.Tensor([theta, ] * R.shape[0])))
with torch.no_grad():
    frag = render(meshes, R=R, T=T)
img_ = to_white_background(frag, texture)

Image.fromarray(torch.min(img_ * 255, torch.full_like(img_, 255)).cpu().numpy().astype(np.uint8)).show()
