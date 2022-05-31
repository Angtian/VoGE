import numpy as np
import torch

from VoGE.Meshes import GaussianMeshes
from pytorch3d.structures import Meshes


def cuboid_gauss(x_range, y_range, z_range, number_vertices, percentage=0.5, colors=None, as_obj=False):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    counts = [yn * xn, yn * xn, (zn - 2) * (xn - 1), (zn - 2) * (xn - 1), (zn - 2) * (yn - 1), (zn - 2) * (yn - 1), ]

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))

    for n in range(1, zn - 1):
        for m in range(xn - 1):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))

    for n in range(1, zn - 1):
        for m in range(1, xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))

    for n in range(1, zn - 1):
        for m in range(1, yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for n in range(1, zn - 1):
        for m in range(yn - 1):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))

    sigma = (edge_length ** 2) / (2 * np.log(1 / percentage)) + 1e-10
    isigma = 1 / sigma

    if colors is not None:
        out_colors = np.concatenate([np.repeat(c[None, :], r, axis=0) for r, c in zip(counts, colors)], axis=0)

        if as_obj:
            return GaussianMeshes(verts=torch.from_numpy(np.array(out_vertices)).type(torch.float32),
                                  sigmas=torch.from_numpy(np.ones(len(out_vertices)) * isigma).type(torch.float32)), out_colors
        else:
            return np.array(out_vertices), np.ones(len(out_vertices)) * isigma, out_colors

    if as_obj:
        return GaussianMeshes(verts=torch.from_numpy(np.array(out_vertices)).type(torch.float32),
                              sigmas=torch.from_numpy(np.ones(len(out_vertices)) * isigma).type(torch.float32))
    else:
        return np.array(out_vertices), np.ones(len(out_vertices)) * isigma


def cuboid_mesh(x_range, y_range, z_range, number_vertices, colors=None, as_obj=False):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    counts = [yn * xn, yn * xn, zn * xn, zn * xn, zn * yn, zn * yn]
    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    if colors is not None:
        out_colors = np.concatenate([np.repeat(c[None, :], r, axis=0) for r, c in zip(counts, colors)], axis=0)

        if as_obj:
            return Meshes(verts=[torch.from_numpy(np.array(out_vertices)).type(torch.float32)],
                          faces=[torch.from_numpy(np.array(out_faces)).type(torch.long)], ), out_colors
        else:
            return np.array(out_vertices), np.array(out_faces), out_colors

    if as_obj:
        return Meshes(verts=[torch.from_numpy(np.array(out_vertices)).type(torch.float32)], 
                      faces=[torch.from_numpy(np.array(out_faces)).type(torch.long)], )
    else:
        return np.array(out_vertices), np.array(out_faces)
        
        
