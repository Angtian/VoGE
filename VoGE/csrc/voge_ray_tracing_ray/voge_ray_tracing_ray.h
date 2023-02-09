#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceVogeRay(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays // (N, 3)
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceVogeRayBackward(
    const at::Tensor& mus,  //  (M, 3, 3)
    const at::Tensor& isigmas, // (M, 3)
    const at::Tensor& rays, // (N, 3)
    const at::Tensor& grad_len,  // (N, M)
    const at::Tensor& grad_act, // (N, M)
    const at::Tensor& grad_dsd // (N, M)
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> FindNearestK(
    const at::Tensor& total_len_in, // [N, M]
    const at::Tensor& total_act_in, // [N, M]
    const at::Tensor& total_dsd_in, // [N, M]
    const float thr_act, // -log(thr + 1e-10)
    const int K
);

#else
    AT_ERROR("Not compiled with GPU support");
#endif