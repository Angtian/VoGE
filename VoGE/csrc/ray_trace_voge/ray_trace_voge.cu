#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>
#include "rasterize_points/rasterization_utils.cuh"


__device__ inline float Innerdot3d(
    const float* target_a, // (P, 3)
    const float* target_b, // (P, 3 * 3)
    const float* target_c, // (P, 3)
    const int p_a,
    const int p_b,
    const int p_c
){
    // A @ B @ C.t : [1, 3] @ [3, 3] @ [3, 1]
    const float a1 = target_a[p_a + 0];
    const float a2 = target_a[p_a + 1];
    const float a3 = target_a[p_a + 2];

    const float b11 = target_b[p_b + 0];
    const float b12 = target_b[p_b + 1];
    const float b13 = target_b[p_b + 2];
    const float b21 = target_b[p_b + 3];
    const float b22 = target_b[p_b + 4];
    const float b23 = target_b[p_b + 5];
    const float b31 = target_b[p_b + 6];
    const float b32 = target_b[p_b + 7];
    const float b33 = target_b[p_b + 8];

    const float c1 = target_c[p_c + 0];
    const float c2 = target_c[p_c + 1];
    const float c3 = target_c[p_c + 2];
    return a1 * b11 * c1 + a1 * b12 * c2 + a1 * b13 * c3 + a2 * b21 * c1 + a2 * b22 * c2 + a2 * b23 * c3 + a3 * b31 * c1 + a3 * b32 * c2 + a3 * b33 * c3;
}


__device__ void inline Innerdot3dBackward(
    const float grad_in,
    const float* target_a, // (P, 3)
    const float* target_b, // (P, 3 * 3)
    const float* target_c, // (P, 3)
    const int p_a,
    const int p_b,
    const int p_c,
    float* grad_a_out,
    float* grad_b_out,
    float* grad_c_out
){
    // g_A = B @ C.t
    // g_B = A.t @ C
    // g_C = A @ B
    const float a1 = target_a[p_a + 0];
    const float a2 = target_a[p_a + 1];
    const float a3 = target_a[p_a + 2];

    const float b11 = target_b[p_b + 0];
    const float b12 = target_b[p_b + 1];
    const float b13 = target_b[p_b + 2];
    const float b21 = target_b[p_b + 3];
    const float b22 = target_b[p_b + 4];
    const float b23 = target_b[p_b + 5];
    const float b31 = target_b[p_b + 6];
    const float b32 = target_b[p_b + 7];
    const float b33 = target_b[p_b + 8];

    const float c1 = target_c[p_c + 0];
    const float c2 = target_c[p_c + 1];
    const float c3 = target_c[p_c + 2];

    atomicAdd(grad_a_out + p_a + 0, (b11 * c1 + b12 * c2 + b13 * c3) * grad_in);
    atomicAdd(grad_a_out + p_a + 1, (b21 * c1 + b22 * c2 + b23 * c3) * grad_in);
    atomicAdd(grad_a_out + p_a + 2, (b31 * c1 + b32 * c2 + b33 * c3) * grad_in);
    
    atomicAdd(grad_b_out + p_b + 0, (a1 * c1) * grad_in); // b11
    atomicAdd(grad_b_out + p_b + 1, (a1 * c2) * grad_in); // b12
    atomicAdd(grad_b_out + p_b + 2, (a1 * c3) * grad_in); // b13
    atomicAdd(grad_b_out + p_b + 3, (a2 * c1) * grad_in); // b21
    atomicAdd(grad_b_out + p_b + 4, (a2 * c2) * grad_in); // b22
    atomicAdd(grad_b_out + p_b + 5, (a2 * c3) * grad_in); // b23
    atomicAdd(grad_b_out + p_b + 6, (a3 * c1) * grad_in); // b31
    atomicAdd(grad_b_out + p_b + 7, (a3 * c2) * grad_in); // b32
    atomicAdd(grad_b_out + p_b + 8, (a3 * c3) * grad_in); // b33
    
    atomicAdd(grad_c_out + p_c + 0, (b11 * a1 + b21 * a2 + b31 * a3) * grad_in);
    atomicAdd(grad_c_out + p_c + 1, (b12 * a1 + b22 * a2 + b32 * a3) * grad_in);
    atomicAdd(grad_c_out + p_c + 2, (b13 * a1 + b23 * a2 + b33 * a3) * grad_in);
}


// TODO: use template
__device__ void inline swap_f(
    float& a,
    float& b
){
    float tem = a;
    a = b;
    b = tem;
}


__device__ void inline swap_i(
    int32_t& a,
    int32_t& b
){
    int32_t tem = a;
    a = b;
    b = tem;
}

__device__ void inline ccp(
    const float* source,
    float* target,
    int32_t size
){
    for (int i = 0; i < size; ++i){
        target[i] = source[i];
    }
}


__device__ void inline vectoratom(
    const float* source,
    float* target,
    int32_t size
){
    for (int i = 0; i < size; ++i){
        atomicAdd(target + i, source[i]);
    }
}

__global__ void RayTraceFineVogeKernel(
    const float* isigmas, // (P, 3, 3)
    const float* mus, // (P, 3)
    const float* rays, // (H, W, 3)
    const int32_t* bin_points, // (BH, BW, T)
    const float thr_act, // -log(thr + 1e-10)
    const int bin_size, 
    const int BH, // num_bins y
    const int BW, // num_bins x
    const int M,
    const int K,
    const int H,
    const int W,
    int32_t* point_idxs, // [W, H, K]
    float* total_len, // [W, H, K]
    float* total_act, // [W, H, K]
    float* total_dsd // [W, H, K]
){
    
    const int num_pixels = BH * BW * bin_size * bin_size;
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < num_pixels; pid += num_threads) {
        // Convert linear index into bin and pixel indices. We make the within
        // block pixel ids move the fastest, so that adjacent threads will fall
        // into the same bin; this should give them coalesced memory reads when
        // they read from points and bin_points.
        int i = pid;
        const int by = i / (BW * bin_size * bin_size); 
        i %= BW * bin_size * bin_size;
        const int bx = i / (bin_size * bin_size);
        i %= bin_size * bin_size;

        // now i is the index inside a bin
        const int yi = i / bin_size + by * bin_size;
        const int xi = i % bin_size + bx * bin_size;

        const int ray_idx = yi * W + xi;

        if (yi >= H || xi >= W)
            continue;

        int32_t current_ptr = 0;
        int32_t tmp_ptr;
        
        for (int m = 0; m < M; ++m) {
            const int idx_point = bin_points[by * BW * M + bx * M + m];

            if (idx_point > -1){
                float k_sig_k = Innerdot3d(rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3);
                float m_sig_k = Innerdot3d(mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3);
                float m_sig_m = Innerdot3d(mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3);

                float hit_length = m_sig_k / k_sig_k;
                float hit_activation = m_sig_m - m_sig_k * m_sig_k / k_sig_k;

                // Top K selection using bubble sort
                // Max time complecity -> O(M * K)
                if (hit_activation < thr_act && hit_length < total_len[ray_idx * K + current_ptr]){
                    total_len[ray_idx * K + current_ptr] = hit_length;
                    total_act[ray_idx * K + current_ptr] = hit_activation;
                    total_dsd[ray_idx * K + current_ptr] = k_sig_k;
                    point_idxs[ray_idx * K + current_ptr] = idx_point;
                    
                    for (tmp_ptr = current_ptr; tmp_ptr > 0 && total_len[ray_idx * K + tmp_ptr] < total_len[ray_idx * K + tmp_ptr - 1]; --tmp_ptr){
                        swap_f(total_len[ray_idx * K + tmp_ptr], total_len[ray_idx * K + tmp_ptr - 1]);
                        swap_f(total_act[ray_idx * K + tmp_ptr], total_act[ray_idx * K + tmp_ptr - 1]);
                        swap_f(total_dsd[ray_idx * K + tmp_ptr], total_dsd[ray_idx * K + tmp_ptr - 1]);
                        swap_i(point_idxs[ray_idx * K + tmp_ptr], point_idxs[ray_idx * K + tmp_ptr - 1]);
                    }

                    if (current_ptr < K - 1){
                        current_ptr++;
                    }
                }
            }
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> RayTraceFineVoge(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays, // (H, W, 3)
    const at::Tensor& bin_points, // (BH, BW, M)
    const float thr_act, // -log(thr + 1e-10)
    const int bin_size,
    const int K
) {
    at::cuda::CUDAGuard device_guard(mus.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // bin_points configuration
    const int BH = bin_points.size(0);
    const int BW = bin_points.size(1);
    const int M = bin_points.size(2);
    const int P = isigmas.size(0);

    const int H = rays.size(0);
    const int W = rays.size(1);

    auto int_opts = bin_points.options().dtype(at::kInt);
    auto float_opts = mus.options().dtype(at::kFloat);

    at::Tensor point_idxs = at::full({H, W, K}, -1, int_opts);
    at::Tensor total_len = at::full({H, W, K}, 1e10, float_opts);
    at::Tensor total_act = at::full({H, W, K}, 1e10, float_opts);
    at::Tensor total_dsd = at::full({H, W, K}, 0, float_opts);
    if (total_len.numel() == 0.0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(point_idxs, total_len, total_act, total_dsd);
    }

    const size_t blocks = 1024;
    const size_t threads = 64;
    
    // const size_t blocks = 32;
    // const size_t threads = 8;

    RayTraceFineVogeKernel<<<blocks, threads, 0, stream>>>(
        isigmas.contiguous().data_ptr<float>(),
        mus.contiguous().data_ptr<float>(),
        rays.contiguous().data_ptr<float>(),
        bin_points.contiguous().data_ptr<int32_t>(),
        thr_act,
        bin_size,
        BH,
        BW,
        M,
        K,
        H,
        W,
        point_idxs.data_ptr<int32_t>(),
        total_len.data_ptr<float>(),
        total_act.data_ptr<float>(),
        total_dsd.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, total_len, total_act, total_dsd);
}


__global__ void RayTraceFineVogeBackwardKernel(
    const float* isigmas, // (P, 3, 3)
    const float* mus, // (P, 3)
    const float* rays, // (H, W, 3)
    const int32_t* point_idxs, // (H, W, k)
    const int K,
    const int H,
    const int W,
    const float* grad_len, // (H, W, K)
    const float* grad_act, // (H, W, K)
    const float* grad_dsd, // (H, W, K, 3)
        
    float* grad_ray, // (H, W, 3)
    float* grad_mus, // (P, 3)
    float* grad_isg  // (P, 3, 3)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < H * W * K; pid += num_threads) {
        const int yi = pid / (W * K);
        const int xi = (pid % (W * K)) / K;

        const int idx_point = point_idxs[pid];

        if (idx_point == -1){
            continue;
        }

        const float grad_l = grad_len[pid];
        const float grad_a = grad_act[pid];
        const float grad_d = grad_dsd[pid];

        const int ray_idx = yi * W + xi;

        float k_sig_k = Innerdot3d(rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3);
        float m_sig_k = Innerdot3d(mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3);
        float m_sig_m = Innerdot3d(mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3);

        const float grad_k_sig_k = (grad_a * m_sig_k - grad_l) * m_sig_k / (k_sig_k * k_sig_k) + grad_d;
        const float grad_m_sig_k = (grad_l - 2 * grad_a * m_sig_k) / k_sig_k;
        const float grad_m_sig_m = grad_a;

        Innerdot3dBackward(grad_k_sig_k, rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3, grad_ray, grad_isg, grad_ray);
        Innerdot3dBackward(grad_m_sig_k, mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3, grad_mus, grad_isg, grad_ray);
        Innerdot3dBackward(grad_m_sig_m, mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3, grad_mus, grad_isg, grad_mus);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceFineVogeBackward(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays, // (H, W, 3)
    const at::Tensor& point_idxs, // (H, W, K)
    const at::Tensor& grad_len,  // (H, W, K)
    const at::Tensor& grad_act, // (H, W, K)
    const at::Tensor& grad_dsd // (H, W, K)
){
    at::cuda::CUDAGuard device_guard(mus.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int H = point_idxs.size(0);
    const int W = point_idxs.size(1);
    const int K = point_idxs.size(2);
    const int P = isigmas.size(0);
    
    auto float_opts = mus.options().dtype(at::kFloat);

    at::Tensor grad_ray_out = at::zeros({H, W, 3}, float_opts);
    at::Tensor grad_mus_out = at::zeros({P, 3}, float_opts);
    at::Tensor grad_isg_out = at::zeros({P, 3, 3}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    RayTraceFineVogeBackwardKernel<<<blocks, threads, 0, stream>>>(
        isigmas.contiguous().data_ptr<float>(),
        mus.contiguous().data_ptr<float>(),
        rays.contiguous().data_ptr<float>(),
        point_idxs.contiguous().data_ptr<int32_t>(),
        K,
        H,
        W,
        grad_len.contiguous().data_ptr<float>(),
        grad_act.contiguous().data_ptr<float>(),
        grad_dsd.contiguous().data_ptr<float>(),
        grad_ray_out.data_ptr<float>(),
        grad_mus_out.data_ptr<float>(),
        grad_isg_out.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_ray_out, grad_mus_out, grad_isg_out);
}

/*
Python implementation of the backward function for proof:

import torch

isigma = torch.Tensor([[2, 1, 0], [1, 1.2, 0], [0, 0, 1]])
mus = torch.Tensor([1.2, 0.2, 0])
rays = torch.Tensor([1, 0, 0])

isigma.requires_grad = True
rays.requires_grad = True
mus.requires_grad = True

msk = mus @ isigma @ rays.T
ksk = rays @ isigma @ rays.T
msm = mus @ isigma @ mus.T

hit_length = msk / ksk
hit_act = msm - msk * msk / ksk

loss = hit_length + hit_act

loss.backward()
print(mus.grad)
print(isigma.grad)
print(rays.grad)


def dot3dback(grad_, a, b, c, g_a, g_b, g_c):
    g_a[0] += (b[0, :] @ c) * grad_
    g_a[1] += (b[1, :] @ c) * grad_
    g_a[2] += (b[2, :] @ c) * grad_

    g_b[0, 0] += (a[0] * c[0]) * grad_
    g_b[0, 1] += (a[0] * c[1]) * grad_
    g_b[0, 2] += (a[0] * c[2]) * grad_
    g_b[1, 0] += (a[1] * c[0]) * grad_
    g_b[1, 1] += (a[1] * c[1]) * grad_
    g_b[1, 2] += (a[1] * c[2]) * grad_
    g_b[2, 0] += (a[2] * c[0]) * grad_
    g_b[2, 1] += (a[2] * c[1]) * grad_
    g_b[2, 2] += (a[2] * c[2]) * grad_

    g_c[0] += (b[:, 0] @ a) * grad_
    g_c[1] += (b[:, 1] @ a) * grad_
    g_c[2] += (b[:, 2] @ a) * grad_

print('--------------------------------')
with torch.no_grad():
    grad_len = 1
    grad_act = 1

    grad_ksk = (grad_act * msk - grad_len) * msk / (ksk * ksk)
    grad_msk = (grad_len - 2 * grad_act * msk) / ksk
    grad_msm = grad_act

    grad_mus = torch.zeros(3)
    grad_isg = torch.zeros(3, 3)
    grad_ray = torch.zeros(3)

    dot3dback(grad_ksk, rays, isigma, rays, grad_ray, grad_isg, grad_ray)
    dot3dback(grad_msk, mus, isigma, rays, grad_mus, grad_isg, grad_ray)
    dot3dback(grad_msm, mus, isigma, mus, grad_mus, grad_isg, grad_mus)

    print(grad_mus)
    print(grad_isg)
    print(grad_ray)
*/
