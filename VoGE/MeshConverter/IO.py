import numpy as np
import torch


"""
GOFF
N_vertices Sigma_Shape(1 or 3 or 6 or 9) If_Radian (1 or 0)
"""


def load_off(file_name, to_torch=False):
    file_handle = open(file_name)

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def load_goff(file_name, to_torch=False):
    file_handle = open(file_name)

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    l_sigma = int(file_list[1].split(' ')[1])
    if_radian = bool(int(file_list[1].split(' ')[2]))

    all_strings = ''.join(file_list[2:2 + n_points])
    points = np.fromstring(all_strings, dtype=np.float32, sep='\n').reshape((-1, 3))

    all_strings = ''.join(file_list[2 + n_points:2 + n_points * 2])
    sigma = np.fromstring(all_strings, dtype=np.float32, sep='\n').reshape((-1, l_sigma))

    if l_sigma == 6:
        sigma = np.split(sigma, (3, 3), axis=1)
    elif l_sigma == 9:
        sigma = sigma.reshape((-1, 3, 3))

    if if_radian:
        all_strings = ''.join(file_list[2 + n_points * 2::])
        radian = np.fromstring(all_strings, dtype=np.float32, sep='\n')
    else:
        radian = None

    if not to_torch:
        return points, sigma, radian
    return torch.from_numpy(points), torch.from_numpy(sigma), torch.from_numpy(radian) if radian is not None else None


def save_off(file_name, vertices, faces):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(file_name, 'w') as fl:
        fl.write(out_string)
    return


def save_goff(file_name, points, sigmas, radians=None):
    if isinstance(sigmas, tuple):
        if isinstance(sigmas[0], torch.Tensor):
            sigmas = torch.cat(sigmas, dim=1)
        else:
            sigmas = np.concatenate(sigmas, axis=1)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.cpu().numpy()
    if isinstance(radians, torch.Tensor):
        radians = radians.cpu().numpy()

    if len(sigmas.shape) > 2:
        sigmas = sigmas.reshape((sigmas.shape[0], -1))
    if len(sigmas.shape) == 1:
        sigmas = np.expand_dims(sigmas, axis=1)
    l_sigma = sigmas.shape[1]
    out_string = 'GOFF\n'
    out_string += '%d %d %d\n' % (points.shape[0], l_sigma, 0 if radians is None else 1)

    for v in points:
        out_string += (('%.16f ' * v.size) % tuple([v_ for v_ in v]))[0:-2] + '\n'

    for v in sigmas:
        out_string += (('%.16f ' * v.size) % tuple([v_ for v_ in v]))[0:-2] + '\n'

    if radians is not None:
        for v in radians:
            out_string += '%.16f' % v + '\n'

    with open(file_name, 'w') as fl:
        fl.write(out_string)
    return


def to_torch(*args):
    return [torch.from_numpy(t).type(torch.float32) if t is not None else None for t in args]


def pre_process_pascal(verts, *args):
    if torch.is_tensor(verts):
        verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    else:
        verts = np.concatenate((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), axis=1)
    return (verts,) + args
