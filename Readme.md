# [VoGE](): A Differentiable Volume Renderer using Neural Gaussian Ellipsoids for Analysis-by-Synthesis

<img src="https://github.com/Angtian/VoGE/blob/main/images/Pipeline.jpg" width="1000"/>

VoGE is a differentiable rendering library conducts differentiable rendering of Gaussian ellipsoids. The rendering process fully follows ray tracing volume densities. VoGE library provide PyTorch APIs for easy usage. For the structure, we refer to [PyTorch3D](https://github.com/facebookresearch/pytorch3d). For the math and details, refer to our [VoGE paper]().

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
| [Change Lighting for Rendering Bunny](https://github.com/Angtian/VoGE/blob/main/demo/ExtractTexture.py) | Rendering PointCloud|

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

We are still working on the documentation.

### [VoGE Renderer](https://github.com/Angtian/VoGE/blob/main/VoGE/Renderer.py)
**class VoGE.Renderer.GaussianRenderer(cameras: pytorch3d.cameras, render_settings: GaussianRenderSettings)(gmeshes, R=None, T=None)**

The VoGE renderer, currently, only support non-batched inputs.

#### Init
*Parameters:*

- cameras: pytorch3d camera (only support PerspectiveCameras).
- render_settings: the render settings, explain below.

#### Call

*Parameters:*

- gmeshes: the Gaussian Ellipsoids set, since it takes equalvalent roles as the mesh in standard mesh rendering process.
- R, T: the cameras pose. None for using the default R and T from self.cameras.

*Returns:*

A fragments contains:
- vert_index \[h, w, M\] gives the indices of the set of nearest effective Gaussian kernels. Invalid will filled with -1. M is max_assign set in GaussianRenderSettings.
- vert_weight \[h, w, M\] indecates the weight of contribution for the corresponded kernel on this pixel.
- valid_num \[h, w\] gives how many Gaussian kernels on that pixel is consider valid.


**class VoGE.Renderer.GaussianRenderSettings(image_size: Union[int, Tuple[int, int]] = 256, max_assign: int = 20, thr_activation: float = 0.01, absorptivity: float = 1, inverse_sigma: bool = False, principal: Union[None, Tuple[int, int], Tuple[float, float]] = None, max_point_per_bin: Union[None, int] = None)**

#### Init
*Parameters:*

- image_size: output image size (h, w).
- max_assign: gives the number of M nearest effective Gaussians traced and returned.
- thr_activation: threshold use in the ray tracing stage, only Gaussians have inital weight > thr are consider effective.
- absorptivity: the absorptivity of the material in volume density aggeration stage, recommand not to change.
- inverse_sigma: if the input sigmas of Gaussians need to be inverse, by default it is already inversed before feed into the renderer.
- principal: the principle point of camera, if None, will use the principle from the cameras.
- max_point_per_bin: the maximum number of Gaussians allowed in each bin in the coarse rasterization stage. max_point_per_bin=-1 removes the coarse rasterization stage. max_point_per_bin=None attempts to set with a heuristic.  

**VoGE.Renderer.interpolate_attr(fragments: Fragments, vert_attr: torch.Tensor)**

Interpolate the render result into a attribute map (gives RGB image when input the RGB colors of each Gaussians).

*Parameters:*

- fragments: fragments returned by the VoGE renderer.
- vert_attr: attribute of each Gaussian kernels, have a shape with (V, C).

**VoGE.Renderer.to_colored_background(fragments: Fragments, colors: torch.Tensor, background_color: Union[torch.Tensor, tuple, list] = (1, 1, 1), thr: float = -1)**

Interpolate the render result into an RGB image with a specific background color.

*Parameters:*

- fragments: fragments returned by the VoGE renderer.
- colors: color of each Gaussian kernels, have a shape with (V, 3).
- background_color: background color, tuple of 3 ints or 1x3 torch.Tensor.
- thr: thr=-1 for soft adding the background color, thr > 0 for a hard mask when adding the background.

**VoGE.Renderer.to_white_background(fragments: Fragments, colors: torch.Tensor, thr: float = -1)**

A specific case for to_colored_background as the background_color=(1, 1, 1).



<!-- ## Understand the Code -->

## Cite













