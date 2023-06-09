#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor> SampleVoge(
    const at::Tensor& image, // (W, H, C)
    const at::Tensor& vert_weight, // (W, H, K)
    const at::Tensor& vert_index,  // (W, H, K)
    const int num_vert
);

std::tuple<at::Tensor, at::Tensor> SampleVogeBackward(
    const at::Tensor& image, // (W, H, C)
    const at::Tensor& vert_weight, // (W, H, K)
    const at::Tensor& vert_index,  // (W, H, K)
    const at::Tensor& grad_feature, // (N, C)
    const at::Tensor& grad_weight_sum // (N, )
);

at::Tensor ScatterMax(
    const at::Tensor& image, // (B, W, H, C)
    const at::Tensor& vert_weight, // (B, W, H, K)
    const at::Tensor& vert_index,  // (B, W, H, K)
    const int num_vert
);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
