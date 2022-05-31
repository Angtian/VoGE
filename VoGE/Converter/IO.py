import numpy as np
import torch


"""
GOFF
N_vertices Sigma_Shape(1 or 3 or 6 or 9) If_Radian (1 or 0)
"""


def load_off(file_name, to_torch=False, ignore_color=False):
    file_handle = open(file_name)

    file_list = file_handle.readlines()

    if ignore_color:
        colored = False
    elif file_list[0][0:3] == 'OFF':
        colored = False
    elif file_list[0][0:4] == 'COFF':
        colored = True
    else:
        raise Exception('Unsupported OFF format: %s' % file_list[0].strip())

    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    verts = np.fromstring(all_strings, dtype=np.float32, sep='\n')
    
    verts = verts.reshape((n_points, -1))

    if colored and verts.shape[1] > 3:
        verts, vert_color = verts[:, 0:3], verts[:, 3::]

        out = [verts, None, vert_color]
    else:
        verts = verts[:, 0:3]
        out = [verts, None]

    n_faces = int(file_list[1].split(' ')[1])
    all_strings = ''.join(file_list[2 + n_points:])
    faces = np.fromstring(all_strings, dtype=np.int32, sep='\n')
    faces = faces.reshape((n_faces, -1))

    n_vert_per_face = int(faces[0][0])

    if colored and faces.shape[1] > faces[0][0] + 1:
        faces, face_color = faces[:, 1:n_vert_per_face + 1], faces[:, (n_vert_per_face + 1)::]

        out[1] = faces
        out += [face_color, ]
    else:
        faces = faces[:, 1:n_vert_per_face + 1]
        out[1] = faces

    if not to_torch:
        return tuple(out)
    else:
        return tuple([torch.from_numpy(t) for t in out])


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


def save_off(file_name, vertices, faces, vert_color=None, face_color=None):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    if vert_color is None and face_color is None:
        out_string = 'OFF\n'
    else:
        out_string = 'COFF\n'

    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    if vert_color is None:
        for v in vertices:
            out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    else:
        if isinstance(vert_color, torch.Tensor):
            vert_color = vert_color.cpu().numpy()
        for v, c in zip(vertices, vert_color):
            out_string += '%.16f %.16f %.16f' % (v[0], v[1], v[2])
            out_string += (' %.16f' * len(c)) % tuple(c)
            out_string += '\n'

    if face_color is None:
        for f in faces:
            out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    else:
        if isinstance(face_color, torch.Tensor):
            face_color = face_color.cpu().numpy()
        for f, c in zip(faces, face_color):
            out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
            out_string += (' %.16f' * len(c)) % tuple(c)
            out_string += '\n'
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
