# Reference: https://pytorch3d.org/tutorials/fit_textured_mesh

import os
import sys
import torch
import pytorch3d

import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    BlendParams
)

from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, interpolate_attr, to_white_background, get_silhouette
from VoGE.Converter import Converters
from VoGE.Meshes import GaussianMeshes

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath(''))

import matplotlib.pyplot as plt

save_img_dir = 'vis_shape_fitting'

os.makedirs(save_img_dir, exist_ok=True)

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


# Setup
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

verts = mesh.verts_packed()
faces = mesh.faces_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))
num_views = 20

# Get a batch of viewing angles.
elev = torch.linspace(0, 360, num_views)
azim = torch.linspace(-180, 180, num_views)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)

cameras = PerspectiveCameras(device=device, R=R, T=T, image_size=((128, 128), ), principal_point=((64, 64), ), focal_length=126, in_ndc=False)
camera = PerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...], image_size=((128, 128), ), principal_point=((64, 64), ), focal_length=126, in_ndc=False)

raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device,
        cameras=camera,
        lights=lights,
        blend_params=BlendParams(background_color=(0, 0, 0))
    )
)

meshes = mesh.extend(num_views)

# Render the cow mesh from each viewing angle
target_images = renderer(meshes, cameras=cameras, lights=lights)

# Our multi-view cow dataset will be represented by these 2 lists of tensors,
# each of length num_views.
target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

sigma = 1e-4
raster_settings_silhouette = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
    faces_per_pixel=50,
)

# Silhouette renderer
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader()
)
silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

def visualize_prediction(predicted_mesh, renderer=renderer_silhouette,
                         target_image=target_rgb[1], title='',
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")

# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")

# We initialize the source shape to be a sphere of radius 1.
src_mesh = ico_sphere(4, device)


def gauss_renderer(gmesh_, R, T, color=None):
    frag_ = render(gmesh_, R=R, T=T)
    
    return torch.cat((interpolate_attr(frag_, color) if color is not None else torch.ones(frag_.vert_weight.shape[0:-1] + (3, )).to(frag_.vert_weight.device), get_silhouette(frag_).unsqueeze(-1)), dim=-1)

render_settings = GaussianRenderSettings(batch_size=-1, image_size=(128, 128), principal=(64, 64), max_assign=25, max_point_per_bin=-1)

camera = PerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...], image_size=((128, 128), ), principal_point=((64, 64), ), focal_length=126)
render = GaussianRenderer(cameras=camera, render_settings=render_settings)


# Number of views to optimize over in each SGD iteration
num_views_per_iteration = 5

# Number of optimization steps
Niter = 2000

# Plot period for the losses
plot_period = 100


gsrc_mesh = Converters.pytorch3d2gaussian(Converters.naive_vertices_converter)(src_mesh, gradianted_args=[True, False, False])
gsrc_mesh = gsrc_mesh.to(device)

vert_color = torch.nn.Parameter(torch.ones((gsrc_mesh.verts.shape[0], 3), device=device) * .5, requires_grad=True)

# The optimizer
optimizer = torch.optim.SGD(list(gsrc_mesh.grad_parameters()) + [vert_color], lr=0.8, momentum=0.9)

losses = {"rgb": {"weight": 0, "values": []},
          "silhouette": {"weight": 1.0, "values": []},
          "edge": {"weight": 1.0, "values": []},
          "normal": {"weight": 0, "values": []},
          "laplacian": {"weight": 0.1, "values": []},
        }

loop = tqdm.trange(Niter)
for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()

    # Losses to smooth /regularize the mesh shape
    loss = {k: torch.tensor(0.0, device=device) for k in losses.keys()}

    all_rgb_loss = []
    all_sil_loss = []
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_predicted = gauss_renderer(gsrc_mesh, R=R[None, j, ...], T=T[None, j, ...], color=vert_color)

        predicted_silhouette = images_predicted[..., 3]
        loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()

        loss["silhouette"] += loss_silhouette / num_views_per_iteration


        predicted_rgb = images_predicted[..., 0:3]
        loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
        loss["rgb"] += loss_rgb / num_views_per_iteration

        all_rgb_loss.append(loss_rgb.item())
        all_sil_loss.append(loss_silhouette.item())

    if i == 400:
        losses['rgb']['weight'] = 1

    sum_loss = torch.tensor(0.0, device=device)
    for k, l in loss.items():
        sum_loss += l * losses[k]["weight"]
        losses[k]["values"].append(float(l.detach().cpu()))

    loop.set_description("total_loss = %.6f" % sum_loss)

    if i % plot_period == 0:
        with torch.no_grad():
            plt.clf()
            gauss_renderer_ = lambda x, R=R[None, 1, ...], T=T[None, 1, ...], color_ = vert_color:gauss_renderer(x, R, T, color=color_)
            visualize_prediction(gsrc_mesh, title="iter: %d" % i, silhouette=False,
                                target_image=target_rgb[1], renderer=gauss_renderer_)
            plt.savefig(save_img_dir + '/%04d.png' % i)

    # Optimization step
    sum_loss.backward()
    optimizer.step()

plt.clf()
gauss_renderer_ = lambda x, R=R[None, 1, ...], T=T[None, 1, ...], color_ = vert_color:gauss_renderer(x, R, T, color=color_)
visualize_prediction(gsrc_mesh, silhouette=False,
                     target_image=target_rgb[1], renderer=gauss_renderer_)
plt.savefig(save_img_dir + '/final.png')
