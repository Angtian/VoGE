import os
import numpy as np
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, to_white_background
from VoGE.Converter import Converters
from VoGE.Meshes import GaussianMeshes
import matplotlib.pyplot as plt


os.system('mkdir data/PittsburghBridge')
os.system('wget data/PittsburghBridge https://dl.fbaipublicfiles.com/pytorch3d/data/PittsburghBridge/pointcloud.npz')


device = 'cuda'

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "PittsburghBridge/pointcloud.npz")

# Load point cloud
pointcloud = np.load(obj_filename)
verts = torch.Tensor(pointcloud['verts'])

verts[:, 1] += 0.5

rgb = torch.Tensor(pointcloud['rgb'][:, 0:3] * 0.85)

cameras = PerspectiveCameras(focal_length=300, principal_point=((160, 160),), image_size=((320, 320),), device=device, )

verts, sigmas, _ = Converters.fixed_pointcloud_converter(verts, radius=0.003, percentage=0.75)

gmesh = GaussianMeshes(verts=verts, sigmas=sigmas).to(device)
rgb = rgb.to(device)
render_settings = GaussianRenderSettings(image_size=(320, 320), principal_point=(160, 160))
renderer = GaussianRenderer(cameras=cameras, render_settings=render_settings)

with torch.no_grad():
    R, T = look_at_view_transform(3.5, 10, 0, device=device)

    frag = renderer(gmesh, R=R, T=T)
    img = to_white_background(frag, rgb).clamp(0, 1)

plt.imshow(img.detach().cpu().numpy())
plt.show()


