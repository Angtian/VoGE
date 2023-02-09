#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>


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


__device__ inline void Innerdot3dBackward(
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


__global__ void RayTraceVogeRayKernel(
    const float* isigmas, // (M, 3, 3)
    const float* mus, // (M, 3)
    const float* rays, // (N, 3)
    const int M,
    const int N,
    float* total_len, // [N, M]
    float* total_act, // [N, M]
    float* total_dsd // [N, M]
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < N * M; pid += num_threads) {
        // Convert linear index into bin and pixel indices. We make the within
        // block pixel ids move the fastest, so that adjacent threads will fall
        // into the same bin; this should give them coalesced memory reads when
        // they read from points and bin_points.
        const int ray_idx = pid / M;
        const int idx_point = pid % M;

        const float k_sig_k = Innerdot3d(rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3);
        const float m_sig_k = Innerdot3d(mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3);
        const float m_sig_m = Innerdot3d(mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3);

        total_len[ray_idx * M + idx_point] = m_sig_k / k_sig_k;
        total_act[ray_idx * M + idx_point] = m_sig_m - m_sig_k * m_sig_k / k_sig_k;
        total_dsd[ray_idx * M + idx_point] = k_sig_k;
    }
}



__global__ void RayTraceVogeRayBackwardKernel(
    const float* isigmas, // (M, 3, 3)
    const float* mus, // (M, 3)
    const float* rays, // (N, 3)
    const int M,
    const int N,
    const float* grad_total_len, // [N, M]
    const float* grad_total_act, // [N, M]
    const float* grad_total_dsd, // [N, M]
    float* grad_isg, // (M, 3, 3)
    float* grad_mus, // (M, 3)
    float* grad_ray // (N, 3)

){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < N * M; pid += num_threads) {
        // Convert linear index into bin and pixel indices. We make the within
        // block pixel ids move the fastest, so that adjacent threads will fall
        // into the same bin; this should give them coalesced memory reads when
        // they read from points and bin_points.
        const int ray_idx = pid / M;
        const int idx_point = pid % M;

        const float grad_l = grad_total_len[pid];
        const float grad_a = grad_total_act[pid];
        const float grad_d = grad_total_dsd[pid];

        const float k_sig_k = Innerdot3d(rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3);
        const float m_sig_k = Innerdot3d(mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3);
        const float m_sig_m = Innerdot3d(mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3);

        const float grad_k_sig_k = (grad_a * m_sig_k - grad_l) * m_sig_k / (k_sig_k * k_sig_k) + grad_d;
        const float grad_m_sig_k = (grad_l - 2 * grad_a * m_sig_k) / k_sig_k;
        const float grad_m_sig_m = grad_a;

        Innerdot3dBackward(grad_k_sig_k, rays, isigmas, rays, ray_idx * 3, idx_point * 9, ray_idx * 3, grad_ray, grad_isg, grad_ray);
        Innerdot3dBackward(grad_m_sig_k, mus, isigmas, rays, idx_point * 3, idx_point * 9, ray_idx * 3, grad_mus, grad_isg, grad_ray);
        Innerdot3dBackward(grad_m_sig_m, mus, isigmas, mus, idx_point * 3, idx_point * 9, idx_point * 3, grad_mus, grad_isg, grad_mus);
    }
}


__global__ void FindNearestKKernel(
    const float* total_len_in, // [N, M]
    const float* total_act_in, // [N, M]
    const float* total_dsd_in, // [N, M]
    const float thr_act, // -log(thr + 1e-10)
    const int M,
    const int K,
    const int N,
    int32_t* point_idxs, // [N, K]
    float* total_len, // [N, K]
    float* total_act, // [N, K]
    float* total_dsd // [N, K]
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < N; pid += num_threads) {
        int32_t current_ptr = 0;
        int32_t tmp_ptr;
        const int ray_idx = pid;
        
        for (int m = 0; m < M; ++m) {
            const float dsd = total_dsd_in[ray_idx * M + m];
            const float hit_activation = total_act_in[ray_idx * M + m];
            const float hit_length = total_len_in[ray_idx * M + m];
            const int idx_point = m;
            
            // Top K selection using bubble sort
            // Max time complecity -> O(M * K)
            if (hit_activation < thr_act && hit_length < total_len[ray_idx * K + current_ptr]){
                total_len[ray_idx * K + current_ptr] = hit_length;
                total_act[ray_idx * K + current_ptr] = hit_activation;
                total_dsd[ray_idx * K + current_ptr] = dsd;
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


std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceVogeRay(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays // (N, 3)
) {
    at::cuda::CUDAGuard device_guard(mus.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // bin_points configuration
    const int M = isigmas.size(0);
    const int N = rays.size(0);

    auto int_opts = mus.options().dtype(at::kInt);
    auto float_opts = mus.options().dtype(at::kFloat);

    at::Tensor total_len = at::full({N, M}, 1e10, float_opts);
    at::Tensor total_act = at::full({N, M}, 0, float_opts);
    at::Tensor total_dsd = at::full({N, M}, 0, float_opts);
    if (total_len.numel() == 0.0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(total_len, total_act, total_dsd);
    }

    const size_t blocks = 1024;
    const size_t threads = 64;
    
    // const size_t blocks = 32;
    // const size_t threads = 8;

    RayTraceVogeRayKernel<<<blocks, threads, 0, stream>>>(
        isigmas.contiguous().data_ptr<float>(),
        mus.contiguous().data_ptr<float>(),
        rays.contiguous().data_ptr<float>(),
        M,
        N,
        total_len.data_ptr<float>(),
        total_act.data_ptr<float>(),
        total_dsd.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(total_len, total_act, total_dsd);
}



std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceVogeRayBackward(
    const at::Tensor& mus,  //  (M, 3, 3)
    const at::Tensor& isigmas, // (M, 3)
    const at::Tensor& rays, // (N, 3)
    const at::Tensor& grad_len,  // (N, M)
    const at::Tensor& grad_act, // (N, M)
    const at::Tensor& grad_dsd // (N, M)
){
    at::cuda::CUDAGuard device_guard(mus.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int M = mus.size(0);
    const int N = rays.size(0);
    
    auto float_opts = mus.options().dtype(at::kFloat);

    at::Tensor grad_ray_out = at::zeros({N, 3}, float_opts);
    at::Tensor grad_mus_out = at::zeros({M, 3}, float_opts);
    at::Tensor grad_isg_out = at::zeros({M, 3, 3}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    RayTraceVogeRayBackwardKernel<<<blocks, threads, 0, stream>>>(
        isigmas.contiguous().data_ptr<float>(),
        mus.contiguous().data_ptr<float>(),
        rays.contiguous().data_ptr<float>(),
        M,
        N,
        grad_len.contiguous().data_ptr<float>(),
        grad_act.contiguous().data_ptr<float>(),
        grad_dsd.contiguous().data_ptr<float>(),
        grad_isg_out.data_ptr<float>(),
        grad_mus_out.data_ptr<float>(),
        grad_ray_out.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_ray_out, grad_mus_out, grad_isg_out);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> FindNearestK(
    const at::Tensor& total_len_in, // [N, M]
    const at::Tensor& total_act_in, // [N, M]
    const at::Tensor& total_dsd_in, // [N, M]
    const float thr_act, // -log(thr + 1e-10)
    const int K
) {
    at::cuda::CUDAGuard device_guard(total_len_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // bin_points configuration
    const int M = total_len_in.size(1);
    const int N = total_len_in.size(0);

    auto int_opts = total_act_in.options().dtype(at::kInt);
    auto float_opts = total_act_in.options().dtype(at::kFloat);

    at::Tensor point_idxs = at::full({N, K}, -1, int_opts);
    at::Tensor total_act = at::full({N, K}, 0, float_opts);
    at::Tensor total_len = at::full({N, K}, 1e10, float_opts);
    at::Tensor total_dsd = at::full({N, K}, 0, float_opts);
    if (total_len.numel() == 0.0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(point_idxs, total_len, total_act, total_dsd);
    }

    const size_t blocks = 512;
    const size_t threads = 32;
    
    // const size_t blocks = 32;
    // const size_t threads = 8;

    FindNearestKKernel<<<blocks, threads, 0, stream>>>(
        total_len_in.contiguous().data_ptr<float>(),
        total_act_in.contiguous().data_ptr<float>(),
        total_dsd_in.contiguous().data_ptr<float>(),
        thr_act,
        M,
        K,
        N,
        point_idxs.data_ptr<int>(),
        total_len.data_ptr<float>(),
        total_act.data_ptr<float>(),
        total_dsd.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, total_len, total_act, total_dsd);
}
