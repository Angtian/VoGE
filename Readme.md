# [VoGE](https://arxiv.org/abs/2205.15401): A Differentiable Volume Renderer using Neural Gaussian Ellipsoids for Analysis-by-Synthesis

<img src="https://github.com/Angtian/VoGE/blob/main/images/Pipeline.jpg" width="1000"/>

VoGE is a differentiable rendering library conducts differentiable rendering of Gaussian ellipsoids. The rendering process fully follows ray tracing volume densities. VoGE library provide PyTorch APIs for easy usage. For the structure, we refer to [PyTorch3D](https://github.com/facebookresearch/pytorch3d). For the math and details, refer to our [VoGE paper](https://arxiv.org/abs/2205.15401).

## Installation
### Requirements
The VoGE library is written in PyTorch, with some components implemented in CUDA for improved performance. Those components also have CPU only substitutions written in PyTorch (will be release soon). A CUDA support GPU with PyTorch-GPU is suggest (currently required) to use VoGE.
- Linux or Windows
- Python 3.8 or 3.9
- PyTorch 1.10
- PyTorch3D 0.6
- Numpy 1.20
- CUDA 10.2 or 11.3

We only list versions of the required packages that have been tested, other version of the described library might also usable.


### Installation
Runing the following code to compile and install VoGE library:
```
git clone https://github.com/Angtian/VoGE.git
cd VoGE
python setup.py install
```
or
```
pip install git+https://github.com/Angtian/VoGE.git
```

Once successfully install VoGE, it should be able to include in your python:
```
import VoGE
```

## Demos

|<img src="https://github.com/Angtian/VoGE/blob/main/images/TextureExtraction.gif" width="320"/>|<img src="https://github.com/Angtian/VoGE/blob/main/images/shape_fitting.gif" width="320"/>|
|-------------------|----------------|
| [Single-viewed Texture Extraction](https://github.com/Angtian/VoGE/blob/main/demo/ExtractTexture.py) | [Shape and Color Fitting using VoGE](https://github.com/Angtian/VoGE/blob/main/demo/ShapeFitting.py)|

|<img src="https://github.com/Angtian/VoGE/blob/main/images/RenderWithLIght.gif" width="320"/>|<img src="https://github.com/Angtian/VoGE/blob/main/images/PointCloud.png" width="320"/>|
|-------------------|----------------|
| [Change Lighting for Rendering Bunny](https://github.com/Angtian/VoGE/blob/main/demo/ExtractTexture.py) | [Rendering PointCloud](https://github.com/Angtian/VoGE/blob/main/demo/RenderPointClouds.py) |

|<img src="https://github.com/Angtian/VoGE/blob/main/images/OcclusionReasoningVoGE.gif" width="320"/>|<img src="https://github.com/Angtian/VoGE/blob/main/images/OcclusionReasoningSoftRas.gif" width="320"/>|
|-------------------|----------------|
| [Using VoGE](https://github.com/Angtian/VoGE/blob/main/demo/ReasonOcclusion.py) | [Baseline: SoftRas](https://github.com/Angtian/VoGE/blob/main/demo/ReasonOcclusionPyTorch3D.py) |
| Single-viewed Occlusion Reasoning -> | for Multi Objects|

|<img src="https://github.com/Angtian/VoGE/blob/main/images/EfficientRepresentation.gif" width="320"/>| ... |
|-------------------|----------------|
| [Efficient Cuboid via Optimization](https://github.com/Angtian/VoGE/blob/main/demo/EfficientCuboidViaOptimization.py) | More |



### In wild object pose Estimation

The in wild object pose estimation experiment using NeMo pipeline will be released in the [NeMo project page](https://github.com/Angtian/NeMo).
|<img src="https://github.com/Angtian/VoGE/blob/main/images/PoseEstimation0.gif" width="320"/> | <img src="https://github.com/Angtian/VoGE/blob/main/images/PoseEstimation1.gif" width="320"/> |
|-------------------|----------------|

## Documentation

### Read the [documentation](https://github.com/Angtian/VoGE/blob/main/Documentation.md).



### Quick Start
Here we give a example to render a cuboid using Gaussian ellispoids:
```
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from VoGE.Converter import Cuboid
from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, to_white_background
import matplotlib.pyplot as plt

device = 'cuda'

# Create gaussians
gaussians = Cuboid.cuboid_gauss((-1, 1), (-1, 1), (-1, 1), 1000, percentage=0.6, as_obj=True).to(device)

# Create a camera
camera = PerspectiveCameras(focal_length=300, image_size=((256, 256), ), principal_point=((128, 128), ), device=device)

# Create the renderer
render_settings = GaussianRenderSettings(image_size=(256, 256), principal=(128, 128), )
renderer = GaussianRenderer(cameras=camera, render_settings=render_settings)

# Compute camera pose
R, T = look_at_view_transform(dist=6, elev=10, azim=70, device=device)

# Render the Gaussians
frag = renderer(gaussians, R=R, T=T)

# Convert into a image
img = to_white_background(frag, (gaussians.verts + 1) / 3).clamp(0, 1)

plt.imshow(img.squeeze(0).detach().cpu().numpy())
plt.show()
```


## Citation
If you find this library is useful, please cite:
```
@article{wang2022voge,
  title={VoGE: A Differentiable Volume Renderer using Gaussian Ellipsoids for Analysis-by-Synthesis},
  author={Wang, Angtian and Wang, Peng and Sun, Jian and Kortylewski, Adam and Yuille, Alan},
  journal={arXiv preprint arXiv:2205.15401},
  year={2022}
}
```













