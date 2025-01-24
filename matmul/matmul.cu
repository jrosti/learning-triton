#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
  

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


__global__
void matmul_naive_kernel(float* A, float* B, float *C, int M, int N, int K) {
    // Matrix product: A x B = C
    // Dimensions:   (M x K) . (K x N) = (M x N)
    
    // faster index over N leads better coalescing
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
 
    float dot_product = 0.0f;
    for (int k=0; k < K; k++) {
        dot_product += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = dot_product;
}


torch::Tensor matmul_naive(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().type() == torch::kCUDA);
    TORCH_CHECK(a.dtype() == torch::kFloat);

    TORCH_CHECK(b.device().type() == torch::kCUDA);
    TORCH_CHECK(b.dtype() == torch::kFloat);

    const auto M = a.size(0);
    const auto N = b.size(1);
    const auto K = a.size(1);
    TORCH_CHECK(b.size(0) == K);

    auto c = torch::zeros({M, N}, a.options());

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(N, threads_per_block.x),
                          cdiv(M, threads_per_block.y));

    matmul_naive_kernel<<<number_of_blocks, threads_per_block, 0>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M,
        N,
        K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
