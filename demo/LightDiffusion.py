from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, interpolate_attr, to_white_background
from VoGE.Utils import rotation_theta
from VoGE.Meshes import GaussianMeshesNaive
from VoGE.Converter.IO import load_goff, load_off, to_torch
from VoGE.Converter.Converters import naive_vertices_converter, normal_mesh_converter
from pytorch3d.renderer.lighting import DirectionalLights
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform, camera_position_from_spherical_angles
from pytorch3d.structures import Meshes
import torch
import torch.nn.functional

import numpy as np
from PIL import Image
import tqdm

device = 'cuda:0'


k = 5

verts_, faces_ = load_off('data/bunny.off', to_torch=True)
mesh_torch = Meshes(verts=[verts_], faces=[faces_])

normals = mesh_torch.verts_normals_packed()
normals = torch.nn.functional.normalize(normals+1e-8, 2).numpy()

meshes = GaussianMeshesNaive(*to_torch(*naive_vertices_converter(verts_.numpy(), faces_.numpy(), percentage=0.6)))
# meshes = GaussianMeshesNaive(*to_torch(*normal_mesh_converter(verts_.numpy(), faces_.numpy(), normals, percentage=0.5, shape_ratio=0.5)))
meshes = meshes.to(device)

render_settings = GaussianRenderSettings(batch_size=-1, image_size=(256, 256), max_assign=40, absorptivity=1, principal=(128, 128), inverse_sigma=False)

normals = Meshes(verts=[verts_], faces=[faces_]).verts_normals_packed()

normals = normals.to(device)

color = torch.ones((1, 3)).to(device)

cameras = PerspectiveCameras(focal_length=2000.0, principal_point=((128, 128),), image_size=(render_settings['image_size'],), device=device, )

renderer = GaussianRenderer(cameras=cameras, render_settings=render_settings)

R, T = look_at_view_transform([6], [0], [10], degrees=True)

cameras.R = R.to(device)
cameras.T = T.to(device)

with torch.no_grad():
    frag = renderer(meshes)

    direction = camera_position_from_spherical_angles(1, elevation=30 + np.abs(100 - k) * 0.5, azimuth=10, device=device)

    normals_map = interpolate_attr(frag, normals)
    light = DirectionalLights(diffuse_color=color, direction=direction)
    output_light = light.diffuse(normals=normals_map)
    img = output_light

Image.fromarray((img * 255).cpu().numpy().astype(np.uint8)).show()
