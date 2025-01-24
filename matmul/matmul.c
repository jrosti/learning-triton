#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float* allocate_matrix(int rows, int cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
    return matrix;
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

void matmul_row_major_column_major(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}


double get_wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);  // Use CLOCK_MONOTONIC for steady wall clock time
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Dimensions
    int M = 4096;  // Rows of A and C
    int K = 16384;  // Columns of A and rows of B
    int N = 1024;  // Columns of B and C

    // Seed for random number generation
    srand(time(NULL));

    // Allocate and initialize matrices
    float* A = allocate_matrix(M, K);
    float* B = allocate_matrix(K, N);
    float* Bt = allocate_matrix(N, K);
    float* C = (float*)malloc(M * N * sizeof(float));

    // Measure execution time
    double start = get_wall_time();
    matmul(A, B, C, M, N, K);
    double end = get_wall_time();

    // Calculate and print execution time
    double elapsed_time = (double)(end - start);
    printf("Matrix multiplication completed in %.6f seconds.\n", elapsed_time);

    start = get_wall_time();
    matmul_row_major_column_major(A, Bt, C, M, N, K);
    end = get_wall_time();

    // Calculate and print execution time
    elapsed_time = (double)(end - start);
    printf("Matrix multiplication row-major completed in %.6f seconds.\n", elapsed_time);

    return 0;
}
