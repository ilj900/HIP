#include <cstdio>
#include <cstdlib>
#include <vector>

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "vector_add.h"

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

bool vector_add() {
    const int N = 1 << 20;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = N - i; }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice);

    const int block = 256;
    vectorAdd<<<(N + block - 1) / block, block>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_c[i] != h_a[i] + h_b[i])
    {
        ok = false;
        break;
    }

    printf("Result: %s\n", ok ? "CORRECT" : "INCORRECT");

    hipFree(d_a); hipFree(d_b); hipFree(d_c);

    return ok;
}