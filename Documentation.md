# Welcome to VoGE's Documentation

We are still working on the documentation.

## [VoGE Renderer](https://github.com/Angtian/VoGE/blob/main/VoGE/Renderer.py)
### class VoGE.Renderer.GaussianRenderer(cameras: pytorch3d.cameras, render_settings: GaussianRenderSettings)(gmeshes, R=None, T=None)

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
- vert_index: (B, H, W, M) gives the indices of the set of nearest effective Gaussian kernels. Invalid will filled with -1. M is max_assign set in GaussianRenderSettings.
- vert_weight: (B, H, W, M) indecates the weight of contribution for the corresponded kernel on this pixel.
- valid_num: (B, H, W) gives how many Gaussian kernels on that pixel is consider valid.


### class VoGE.Renderer.GaussianRenderSettings(image_size: Union[int, Tuple[int, int]] = 256, max_assign: int = 20, thr_activation: float = 0.01, absorptivity: float = 1, inverse_sigma: bool = False, principal: Union[None, Tuple[int, int], Tuple[float, float]] = None, max_point_per_bin: Union[None, int] = None)

#### Init
*Parameters:*

- image_size: output image size (H, W).
- max_assign: gives the number of M nearest effective Gaussians traced and returned.
- thr_activation: threshold use in the ray tracing stage, only Gaussians have inital weight > thr are consider effective.
- absorptivity: the absorptivity of the material in volume density aggeration stage, recommand not to change.
- inverse_sigma: if the input sigmas of Gaussians need to be inverse, by default it is already inversed before feed into the renderer.
- principal: the principle point of camera, if None, will use the principle from the cameras.
- max_point_per_bin: the maximum number of Gaussians allowed in each bin in the coarse rasterization stage. max_point_per_bin=-1 removes the coarse rasterization stage. max_point_per_bin=None attempts to set with a heuristic.  

### VoGE.Renderer.interpolate_attr(fragments: Fragments, vert_attr: torch.Tensor)

Interpolate the render result into a attribute map (gives RGB image when input the RGB colors of each Gaussians).

*Parameters:*

- fragments: fragments returned by the VoGE renderer.
- vert_attr: attribute of each Gaussian kernels, have a shape with (N, C).

*Returns:*

- attribute map: torch.Tensor with shape (B, H, W, C)

### VoGE.Renderer.to_colored_background(fragments: Fragments, colors: torch.Tensor, background_color: Union[torch.Tensor, tuple, list] = (1, 1, 1), thr: float = -1)

Interpolate the render result into an RGB image with a specific background color.

*Parameters:*

- fragments: fragments returned by the VoGE renderer.
- colors: color of each Gaussian kernels, have a shape with (N, 3).
- background_color: background color, tuple of 3 ints or 1x3 torch.Tensor.
- thr: thr=-1 for soft adding the background color, thr > 0 for a hard mask when adding the background.

*Returns:*

- RGB image: torch.Tensor with shape (B, H, W, 3)

### VoGE.Renderer.to_white_background(fragments: Fragments, colors: torch.Tensor, thr: float = -1)

A specific case for to_colored_background as the background_color=(1, 1, 1).


*Parameters:*

- fragments: fragments returned by the VoGE renderer.
- colors: color of each Gaussian kernels, have a shape with (N, 3).
- thr: thr=-1 for soft adding the background color, thr > 0 for a hard mask when adding the background.

*Returns:*

- RGB image: torch.Tensor with shape (B, H, W, 3)


## [VoGE Sampler](https://github.com/Angtian/VoGE/blob/main/VoGE/Sampler.py)

VoGE sampler conducts the inverse process of rendering. VoGE sampler convert an input attribute map (feature map or rgb image) into attribute for each Gaussian kernels.

### VoGE.Sampler.sample_features(frag, image, n_vert=None)

The CUDA implementation of VoGE sampling layer (differentiable). It is eqaulvalent to this pytorch code:
```
    >>> weight = torch.zeros(image.shape[0:2] + (n_vert, ))
    >>> weight = ind_fill(weight, frag.vert_index, dim=2, src=frag.vert_weight)
    >>> sum_weight = torch.sum(weight, dim=(0, 1), keepdim=True)
    >>> features = weight.view(-1, weight.shape[-1]).T @ image.view(-1, 3) # [H * W, N].T @ [H * W, C] -> [N, C]
```
Note the pytorch code consumes lot of memory especially when the number of Gaussians is large, while our CUDA version is very efficient.

*Parameters:*

- frag: fragments returned by the VoGE renderer.
- image: torch.Tensor, (B, H, W, C), image or feature map.
- n_vert: int, number of Gaussian ellipsoids. Default: use the max of frag.vert_index.

*Returns:*

- features: torch.Tensor, (N, C), color or feature for each Gaussian.
- sum_weight: torch.Tensor, (N, ), accumlative weight sum for each Gaussian.

**Important:** 
The output features are, by default, not normalized (multiplied by sum_weight). To get the normalized color:
```
features = features / sum_weight[..., None]
```



## [Gaussian Ellipsoids](https://github.com/Angtian/VoGE/blob/main/VoGE/Meshes.py)
Since the Gaussian ellipsoids takes the equalvalent role as Meshes in the PyTorch3D meshes renderer, for convenience, here we name them GaussianMeshes, though they are not real meshes.

### class VoGE.Meshes.GaussianMeshesNaive(verts, sigmas)

The hook version of Gaussian Ellipsoids (do not store parameters). The class will return the reference of original verts and sigmas, so the gradient will compute toward the input in initalization.

#### Init

*Parameters:*

- verts: torch.Tensor, (N, 3), center of Gaussian Ellipsoids.
- sigmas: torch.Tensor, (N, ) or (N, 3, 3), spacial varience of the Gaussian Ellipsoids, 1-dims tensor for isotropic Gaussian, 3-dims tensor for anisotropic.

#### Call

*Returns:*

- verts: reference of the input verts in initalization.
- sigmas: reference of the input sigmas in initalization.


### class VoGE.Meshes.GaussianMeshes(verts, sigmas)

The nn.Module version of Gaussian Ellipsoids. The verts and sigmas will be copied and stored as nn.Parameter. Gradient will compute toward parameters store inside this class.

#### Init

*Parameters:*

- verts: torch.Tensor, (N, 3), center of Gaussian Ellipsoids.
- sigmas: torch.Tensor, (N, ) or (N, 3, 3), spacial varience of the Gaussian Ellipsoids, 1-dims tensor for isotropic Gaussian, 3-dims tensor for anisotropic.

#### Call

*Returns:*

- verts: inside parameter verts.
- sigmas: inside parameter sigmas.

## [VoGE Converters](https://github.com/Angtian/VoGE/blob/main/VoGE/Converter/Converters.py)

VoGE Converter converts existing 3D representation (meshes or pointclouds) into Gaussian ellipsoids. We will add more converts in future.

### VoGE.Converter.Converters.naive_vertices_converter(vertices, faces, percentage=0.5, max_sig_rate=-1)

A naive mesh converter, which returns each Gaussian is correspond to each vertices of the mesh. The center of each gaussian is place at the location of corresponded vertex. Each Gaussian is isotropic (but different Gaussian can have the different sigmas). Each sigma is compute with a function of distances from its corresponded vertex to adjacent vertices. Normally, the source mesh should be uniformly sampled.

*Parameters:*

- vertices: np.ndarray or torch.Tensor(cpu), (N, 3), vertices locations of the source mesh.
- faces: np.ndarray or torch.Tensor(cpu), (F, 3), faces of the source mesh.
- percentage: float, a hyper-parameter in the function for computing sigma. Larger gives larger spacial variance of each Gaussian.
- max_sig_rate: float, limit the maximum spacial variance to not be large than max_sig_rate of the average value. max_sig_rate=-1 for no limits.


*Returns:*

- verts: np.ndarray or torch.Tensor(cpu), (N, 3), center of each Gaussian
- sigmas: np.ndarray or torch.Tensor(cpu), (N, ), spacial variance of each Gaussian
- None

### VoGE.Converter.Converters.normal_mesh_converter(vertices, faces, normals, percentage=0.5, shape_ratio=0.5, max_sig_rate=-1, auto_fix=True)

A mesh converter, similar to naive_vertices_converter, but give flattern anisotropic Gaussians along the normal direction for each mesh vertices.

*Parameters:*

- vertices: np.ndarray or torch.Tensor(cpu), (N, 3), vertices locations of the source mesh.
- faces: np.ndarray or torch.Tensor(cpu), (F, 3), faces of the source mesh.
- percentage: float, a hyper-parameter in the function for computing sigma. Larger gives larger spacial variance of each Gaussian.
- shape_ratio: float, determine how flattern each output Gaussian is. Smaller makes more flattern.
- max_sig_rate: float, limit the maximum spacial variance to not be large than max_sig_rate of the average value. max_sig_rate=-1 for no limits.
- auto_fix: Avoid potiential error for uninversable matrix keep True.


*Returns:*

- verts: np.ndarray or torch.Tensor(cpu), (N, 3), center of each Gaussian
- sigmas: np.ndarray or torch.Tensor(cpu), (N, 3, 3), spacial variance of each Gaussian
- None

### VoGE.Converter.Converters.naive_point_cloud_converter(points, percentage=0.5, n_nearest=4, thr_max=2)
A naive pointcloud converter, which returns each Gaussian is correspond to each point respectively. The spacial varience compute as a function of the distance from that point to n_nearest neighbors.

*Parameters:*

- points: np.ndarray or torch.Tensor(cpu), (N, 3), points locations of the source mesh.
- percentage: float, a hyper-parameter in the function for computing sigma. Larger gives larger spacial variance of each Gaussian.
- n_nearest: int, number of nearest points considered during the sigmas computation.
- thr_max: float, limit the maximum spacial variance to not be large than thr_max of the average value.

*Returns:*

- verts: np.ndarray or torch.Tensor(cpu), (N, 3), center of each Gaussian
- sigmas: np.ndarray or torch.Tensor(cpu), (N, 3, 3), spacial variance of each Gaussian
- None

### VoGE.Converter.Converters.fixed_pointcloud_converter(points, radius, percentage=0.5)
A naive pointcloud converter, which returns each Gaussian is correspond to each point respectively. The spacial varience compute as a function of the input radius.

*Parameters:*

- points: np.ndarray or torch.Tensor(cpu), (N, 3), points locations of the source mesh.
- radius: float or torch.Tensor(cpu), (N, ), the radius of each point.
- percentage: float, a hyper-parameter in the function for computing sigma. Larger gives larger spacial variance of each Gaussian.

*Returns:*

- verts: np.ndarray or torch.Tensor(cpu), (N, 3), center of each Gaussian
- sigmas: np.ndarray or torch.Tensor(cpu), (N, 3, 3), spacial variance of each Gaussian
- None


### VoGE.Converter.Converters.pytorch3d2gaussian(converter, **kwargs)

A function decorator allows input of converters as pytorch3d.Meshes and output as VoGE.GaussianMeshes.

*Parameters:*
- converter: python function, the original converter
- kwargs: args feed into the converter

*Example:*
```
convert_function = Converters.pytorch3d2gaussian(Converters.naive_vertices_converter, percentage=0.7)
gmesh = convert_function(meshes)

convert_function = Converters.pytorch3d2gaussian(Converters.fixed_pointcloud_converter, radius=0.2)
gmesh = convert_function(pointclouds)
```



