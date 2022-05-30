# [VoGE](): A Differentiable Volume Renderer using Neural Gaussian Ellipsoids for Analysis-by-Synthesis

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

## To Run VoGE

## Demos

## How does VoGE works

## Cite













