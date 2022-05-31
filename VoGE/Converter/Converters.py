import numpy as np
import os
import torch
from VoGE.Meshes import GaussianMeshes
from pytorch3d.renderer import look_at_rotation
from VoGE.Utils import Batchifier


def get_vert_edge_length(verts, faces, default_l=1e-3):
    edges = np.ones((verts.shape[0], 60)) * -1
    edge_count = np.zeros(verts.shape[0], dtype=np.int32)

    for f in faces:
        for v in f:
            edges[v, edge_count[v]:edge_count[v]+3] = f[0:3]
            edge_count[v] += 3

    len_vertices = np.ones(verts.shape[0]) * default_l

    for i, _ in enumerate(edges):
        if edge_count[i] == 0:
            continue

        idxs = np.unique(edges[i, 0:edge_count[i]]).astype(np.int32)

        v_ = verts[i]    # [3, ]
        vs = verts[idxs] # [k, 3]
        len_sum = np.sum(((vs - v_[None, :]) ** 2).sum(1) ** .5)
        len_vertices[i] = len_sum / (idxs.shape[0] - 1)

    return len_vertices


def normal_mesh_converter(vertices, faces, normals, percentage=0.5, shape_ratio=0.5, max_sig_rate=-1, auto_fix=True):
    if torch.is_tensor(vertices):
        vertices = vertices.numpy()
        faces = faces.numpy()
        is_torch = True
    else:
        normals = torch.from_numpy(normals)
        is_torch = False

    default_l = 10 * np.sum((vertices.max(axis=0) - vertices.min(axis=0)) ** 2) ** 0.5 / vertices.shape[0]
    average_len = get_vert_edge_length(vertices, faces, default_l)

    isigma_base = 1 / ((average_len ** 2) / (2 * np.log(1 / percentage)) + 1e-10)

    assert torch.max((normals ** 2).sum(-1)) < 1.1
    assert torch.min((normals ** 2).sum(-1)) > 0.9

    # [n, 3, 3]
    base_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, shape_ratio]])[None, ...] * isigma_base.reshape((-1, 1, 1))

    # [n, 3, 3]
    rotations_matrix = look_at_rotation(-normals).numpy()

    rotations_matrix = rotations_matrix
    isigma = rotations_matrix @ base_ @ rotations_matrix.transpose(0, 2, 1)

    if auto_fix:
        dets = np.linalg.det(isigma)
        isigma[dets == 0] = np.eye(3)[None, ...] * isigma_base[dets == 0].reshape((-1, 1, 1))

    if max_sig_rate > 0:
        thr = np.mean(isigma) * max_sig_rate
        isigma[isigma > thr] = thr
    if is_torch:
        return torch.from_numpy(vertices).type(torch.float32), torch.from_numpy(isigma).type(torch.float32), None
    else:
        return vertices, isigma, None


def naive_vertices_converter(vertices, faces, percentage=0.5, max_sig_rate=-1):
    if torch.is_tensor(vertices):
        vertices = vertices.numpy()
        faces = faces.numpy()
        is_torch = True
    else:
        is_torch = False

    default_l = 10 * np.sum((vertices.max(axis=0) - vertices.min(axis=0)) ** 2) ** 0.5 / vertices.shape[0]
    average_len = get_vert_edge_length(vertices, faces, default_l)

    sigma = (average_len ** 2) / (2 * np.log(1 / percentage)) + 1e-10
    isigma = 1 / sigma

    if max_sig_rate > 0:
        thr = np.mean(isigma) * max_sig_rate
        isigma[isigma > thr] = thr

    if is_torch:
        return torch.from_numpy(vertices).type(torch.float32), torch.from_numpy(isigma).type(torch.float32), None
    else:
        return vertices, isigma, None


def naive_point_cloud_converter(points, percentage=0.5, n_nearest=4, thr_max=2):
    if not torch.is_tensor(points):
        points = torch.from_numpy(points)
        to_np = True
    else:
        to_np = False
    points = points.type(torch.float32)

    def foo(point_v, point_t):
        point_dist = (point_v - point_t).pow(2).sum(-1).pow(.5)
        top_dist = torch.topk(point_dist, k=n_nearest, dim=1, largest=False)[0]
        average_len = torch.min(top_dist, top_dist.mean(dim=1, keepdim=True).expand(-1, n_nearest) * thr_max).mean(dim=1)

        return (average_len ** 2) / (4 * np.log(1 / percentage))

    if points.shape[0] > 1e5:
        foo = Batchifier(int(1e9 / points.shape[0]), batch_args='point_v', target_dims=0, tbar=True)(foo)
    with torch.no_grad():
        sigma = foo(point_v=points.unsqueeze(1), point_t=points.unsqueeze(0)) + 1e-8
    isigma = 1 / sigma

    if to_np:
        return points.numpy(), isigma.numpy(), None
    else:
        return points, isigma, None


def fixed_pointcloud_converter(points, radius, percentage=0.5):
    if not torch.is_tensor(points):
        points = torch.from_numpy(points)
        if not isinstance(radius, float):
            radius = torch.from_numpy(radius)
        to_np = True
    else:
        to_np = False

    isigma = torch.ones(points.shape[0]) / ((radius ** 2) / (2 * np.log(1 / percentage)) + 1e-10)

    if to_np:
        return points.numpy(), isigma.numpy(), None
    else:
        return points, isigma, None   
    

def convert_path(source_path, destiny_path, convert_function, filter_=None):
    this_fl_list = os.listdir(source_path)
    os.makedirs(destiny_path, exist_ok=True)

    for this_name in this_fl_list:
        this_source_path = os.path.join(source_path, this_name)
        this_destiny_path = os.path.join(destiny_path, this_name)

        if os.path.isfile(this_source_path):
            if filter_ is not None and not filter_(this_name):
                continue
            convert_function(this_source_path, this_destiny_path)
        else:
            convert_path(this_source_path, this_destiny_path, convert_function)


class ComposedConverter(object):
    def __init__(self, loader, saver, converter, **kwargs):
        super().__init__()
        self.loader = loader
        self.saver = saver
        self.converter = converter
        self.kwargs = kwargs

    def __call__(self, source_path, destiny_path):
        get = self.loader(source_path)
        if not isinstance(get, tuple):
            get = (get, )
        get = self.converter(*get, **self.kwargs)
        if not isinstance(get, tuple):
            get = (get, )
        self.saver(destiny_path, *get)


def pytorch3d2gaussian(converter, **kwargs):
    def wrapper(input_, **mesh_kwargs):
        if isinstance(input_, Meshes):
            mesh = input_
            if len(mesh) > 1:
                mesh = mesh[0]

            verts = mesh.verts_packed().cpu()
            faces = mesh.faces_packed().cpu()

            verts, sigmas, radians = converter(verts, faces, **kwargs)
        elif isinstance(input_, Pointclouds):
            pointscloud = input_
            points = pointscloud.points_packed()

            verts, sigmas, radians = converter(points, **kwargs)

        return GaussianMeshes(verts.type(torch.float32), sigmas.type(torch.float32), radians.type(torch.float32) if radians is not None else None, **mesh_kwargs).to(input_.device)
    return wrapper

