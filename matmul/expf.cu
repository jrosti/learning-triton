#include <stdio.h>
#include <cuda_runtime.h>

// Define the CUDA kernel
__global__ void expKernel(float *d_out, const float *d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx]*d_in[idx];
    }
}

int main() {
    const int N = 1000000000;  // Large N
    float *h_in, *h_out;     // Host pointers
    float *d_in, *d_out;     // Device pointers

    // Allocate large arrays on the heap
    h_in = (float*)malloc(N * sizeof(float));
    h_out = (float*)malloc(N * sizeof(float));

    if (!h_in || !h_out) {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i / 1000000.0f;  // Some values to test expf
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    expKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);

    // Copy results from device to host
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few results for verification
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f -> %f\n", h_in[i], h_out[i]);
    }

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
