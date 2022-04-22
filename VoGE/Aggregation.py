import torch
import torch.nn.functional as F
import math
from VoGE.Utils import ind_sel, ind_fill


def inverse_cumsum(x, dim):
    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


def get_ray_camera_space(img_size, principle, focal, device='cpu'):
    if isinstance(focal, float) or isinstance(focal, int):
        focal = torch.ones(2, device=device) * focal
    elif focal.dim() == 2:
        focal = focal.squeeze()
    elif focal.shape[0] == 1:
        focal = focal.expand(2)
    h, w, = img_size

    i, j = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w))
    i = i.to(device)
    j = j.to(device)

    # object coordinate: [x, y, z]; image coordinate [y, x]
    dirs = torch.stack([- (j - principle[1]) / focal[1], - (i - principle[0]) / focal[0], torch.ones_like(i)], -1)

    return F.normalize(dirs, p=2, dim=2)


def get_cross_activation(sel_length, sel_dsd):
    """
        target:
        integrate {-inf -> l_k} exp(-(l_m * K - Mu_k).T @ iSigma_k @ (l_m * K - Mu_k))
                = exp(-(l_k * K - Mu_k).T @ iSigma_k @ (l_k * K - Mu_k)) *
                  integrate {-inf -> 0} exp( (l_m - l_k) ^ 2 * K.T @ iSigma_k @ K )
                = exp(-act_k) * erf( (l_m - l_k) * (K.T @ iSigma_k @ K) ^ .5 )
        here we calculate
        (l_m - l_k) * (K.T @ iSigma_k @ K) ^ .5
    :param ray_grids: [k, 3]
    :param sel_length: [k, M]
    :param sel_dsd: [k, M] -> 2 * K.T @ iSigma_k @ K
    :return: cross_activation [k, M (_m), M (_k)]:
                     (l_m - l_k) * (K.T @ iSigma_k @ K) ^ .5
    """
    k, m = sel_length.shape

    # [k, M(_m), 1] - [k, 1, M(_k)] -> [k, M(_m), M(_k)]
    # [k, M(_m), M(_k)] * [k, 1, M(_k)]
    cross_activation = (sel_length.unsqueeze(2) - sel_length.unsqueeze(1)) * (sel_dsd.view(k, 1, m) + 1e-10).pow(.5) # - (1 + sel_radian.unsqueeze(1)).pow(0.5) # (1 + sel_radian.unsqueeze(1))

    return cross_activation


def assign2weight(sel_activation, cross_activation, occupation_weight=1.):
    """
    Final output of whole model (M=num_assign, r=hit_distance, l=hit_length), return weight at hit point:

    Sum_{m=1}^M f_m * exp(- Sum_{k=1}^(M, m != k) (exp(-act_k) * (erf( (l_m - l_k) * (K.T @ iSigma_k @ K) ^ .5 ) + 1)) / 2 ) * exp(- (l_m * K - Mu_m).T @ iSigma @ (l_m * K - Mu_m) )

    solid_foo -> (exp(-act_k) * [ (erf( ca ) + 1)) / 2 + (relu(ca + r') - relu(ca - r'))] | ca = (l_m - l_k) * (K.T @ iSigma_k @ K) ^ .5

    :param sel_activation: [k, M]
    :param cross_activation: [k, M, M]
    :param occupation_weight: float, the weight when calculating mutual occlusion.

    :return: (weight)
    weight: [k, M]
    """
    # [k, 1, M(_k)] * erf( [k, M (_m), M (_k)] )
    density_dist = torch.exp(-sel_activation.unsqueeze(1)) * ( (torch.erf(cross_activation) + 1) / 2)

    # Sum_{k=1}^(M) -> Sum_{k=1}^(M, m != k)
    # [k, M (_m), M (_k)].sum(2) - [k, M (_m)]
    density_weight = torch.exp(-(torch.sum(density_dist, dim=2)) * occupation_weight)

    # [k, M] * exp([k, M])
    weight = density_weight * torch.exp(-sel_activation)

    return weight / math.exp(-0.5)


def aggregation(sel_idx: torch.Tensor, sel_act: torch.Tensor, sel_len: torch.Tensor, sel_dsd: torch.Tensor, occupation_weight: float=1.):
    """
    :param sel_idx: [w, h, M] vertices indexs along rays
    :param sel_act: [w, h, M] hit activations along rays
    :param sel_len: [w, h, M] hit length along rays
    :param sel_dsd: [w, h, M] eaualvalent isigma along the ray -> D.T @ isigmas @ D
    :param occupation_weight: float, the weight when calculating mutual occlusion. In the base function ( exp(o_weight * x) ) of when calculating T(tor(x))
    :return: weight matrix: [n(optional), w, h, M]: torch.float32 vertices weight at each location
             index matrix: [n(optional), w, h, M]: torch.int
             valid_num: [n(optional), w, h]: torch.int
    """
    M = sel_idx.shape[-1]

    shape_ = sel_idx.shape[0:-1]
    reshape_foo = lambda x, tar_shape=shape_: x.view(*tar_shape, *dict() if len(x.shape) == 1 else (-1, ))

    # [k', M(_m), M(_k)]
    cross_activation = get_cross_activation(sel_length=sel_len.view(-1, M), sel_dsd=sel_dsd.view(-1, M))

    # [k', M]
    get_weight = assign2weight(sel_activation=sel_act.view(-1, M), cross_activation=cross_activation, occupation_weight=occupation_weight)

    valid_num = torch.sum(sel_idx >= 0, dim=-1)

    # [k', M] -> [b, w, h, M]
    return reshape_foo(get_weight), sel_idx, valid_num, sel_len



def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor, valid_num: torch.Tensor):
    """

    :param vert_attr: [n, d] color or feature of each vertex
    :param weight: [b(optional), w, h, M] weight of selected vertices
    :param vert_assign: [b(optional), w, h, M] selective index
    :param valid_num: [b(optional), w, h, ]
    :return:
    """
    assert vert_attr.shape[0] > vert_assign.max()
    with torch.no_grad():
        target_dim = len(valid_num.shape)

        # [b(optional), w, h, M]
        mask = torch.zeros(weight.shape[0:-1] + (weight.shape[-1] + 1, ), device=weight.device, dtype=weight.dtype)
        mask = ind_fill(mask, valid_num.unsqueeze(target_dim).type(torch.long), dim=target_dim, src=1)

        # first num filled with 1, later 0. => [1, ] * valid_num + [0, ] * (M - valid_num)
        mask = inverse_cumsum(mask, dim=target_dim)[..., 1:]

        vert_assign += (vert_assign < 0) * 1

    # [b(optional), w, h, M]
    weight = mask * weight

    # [n, d] ind: [b(optional), w, h, M]-> [b(optional), w, h, M, d]
    sel_attr = ind_sel(vert_attr[(None, ) * target_dim], vert_assign.type(torch.long), dim=target_dim)

    # [b(optional), w, h, M]
    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def expend_sigma(sigma, rotation_matrix=None):
    """
    Return rotated sigma:
            [[ sigma[0] * R[0, 0], sigma[1] * R[0, 1], sigma[2] * R[0, 2] ],
             [ sigma[0] * R[1, 0], sigma[1] * R[1, 1], sigma[1] * R[1, 2] ],
             [ sigma[0] * R[2, 0], sigma[1] * R[2, 1], sigma[1] * R[2, 2] ]]
    :param sigma: [n] or [n, 3] or [n, 3, 3]
    :param rotation_matrix: Optional [n, 3, 3]
    :return:
    """
    if len(sigma.shape) == 3:
        if sigma.shape[1] == 3 and sigma.shape[2] == 3:
            return sigma
        else:
            raise Exception('Got unexpected sigma, which has shape: ' + str(sigma.shape))

    if rotation_matrix is None:
        # [1, 3, 3]
        rotation_matrix = torch.eye(3, device=sigma.device).unsqueeze(0)

    if len(rotation_matrix.shape) == 2:
        rotation_matrix.unsqueeze(0)

    rotation_matrix = rotation_matrix[:, :3, :3]

    if len(sigma.shape) == 1:
        return sigma.unsqueeze(1).unsqueeze(2) * rotation_matrix

    if len(sigma.shape) == 2:
        return sigma.unsqueeze(2) * rotation_matrix

    raise Exception('Got unexpected sigma, which has shape: ' + str(sigma.shape))
