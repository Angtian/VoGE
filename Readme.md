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
Other version of the described library might also usable.


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

## Documentation

## Understand the Code

## Cite













