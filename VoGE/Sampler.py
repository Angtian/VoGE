from VoGE import _C
import torch


def sample_features(frag, image, n_vert=None):
    """
    Conduct the feature extractions. Same as python code:
    >>> weight = torch.zeros(image.shape[0:2] + (n_vert, ))
    >>> weight = ind_fill(weight, frag.vert_index, dim=2, src=frag.vert_weight)
    >>> vert_sum_weight = torch.sum(weight, dim=(0, 1), keepdim=True)
    >>> vert_feature = weight.view(-1, weight.shape[-1]).T @ image.view(-1, 3) # [H * W, N].T @ [H * W, C] -> [N, C]
    :param frag: fragment
    :param image: [w, h, c] image
    :param n_vert: number of vert, default: max(vert_index)
    :return: vert_feature: [n, c]
             vert_sum_weight: [n, ] sum of all sampling weight, for normalization
    """
    vert_weight = frag.vert_weight
    vert_index = frag.vert_index

    if n_vert is None:
        if hasattr(frag, 'num_vertices'):
            n_vert = frag.num_vertices
        else:
            n_vert = vert_index.max() + 1
    assert image.device == vert_index.device
    assert n_vert > vert_index.max()
    assert vert_weight.shape[0] == image.shape[0] and vert_weight.shape[1] == image.shape[1]
    return _SampleVoGE.apply(image, vert_weight, vert_index, n_vert)


class _SampleVoGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                image,
                vert_weight,
                vert_index,
                num_vert
        ):
        args = (
                image,  # (H, W, C)
                vert_weight,  # (H, W, K)
                vert_index,  # (H, W, K)
                num_vert
        )
        vert_feature, vert_sum_weight = _C.sample_voge(*args)
        ctx.save_for_backward(image, vert_weight, vert_index)
        return vert_feature, vert_sum_weight

    @staticmethod
    def backward(ctx,
                 grad_vert_feature,
                 grad_vert_weight_sum
        ):
        image, vert_weight, vert_index = ctx.saved_tensors
        args = (
            image,
            vert_weight,
            vert_index,
            grad_vert_feature,
            grad_vert_weight_sum
        )
        grad_image, grad_weight = _C.sample_voge_backward(*args)
        return grad_image, grad_weight, None, None

