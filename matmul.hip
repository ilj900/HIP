#include <hip/hip_runtime.h>
#include "gtest/gtest.h"

#include <random>
#include <vector>

#include "matmul.h"

__global__ void matmul1d(const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    const uint Size = M * N;

    if (I >= Size)
    {
        return;
    }

    int Row = I / N;
    int Col = I % N;

    float Acc = 0;
    for (int i = 0; i < K; ++i)
    {
        float L = A[Row * K + i];
        float R = B[Col + (K * i)];
        Acc += L * R;
    }
    C[I] = Acc;
}

void MatMul1D(uint32_t M, uint32_t N, uint32_t K, uint32_t S)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    uint32_t Size = C.size();

    std::mt19937 Engine{0};
    std::uniform_real_distribution<float> Dist(0.0f, 20.0f);
    auto Gen = [&Engine, &Dist]() -> float
    {
        return Dist(Engine);
    };

    for (uint32_t i = 0; i < M * K; ++i)
    {
        A[i] = Gen();
    }

    for (uint32_t i = 0; i < K * N; ++i)
    {
        B[i] = Gen();
    }

    float* HipA;
    float* HipB;
    float* HipC;
    ASSERT_EQ(hipMalloc(&HipA, A.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&HipB, B.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&HipC, C.size() * sizeof(float)), hipSuccess);

    ASSERT_EQ(hipMemcpy(HipA, A.data(), A.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(HipB, B.data(), B.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

    matmul1d<<<(Size + S - 1) / S, S>>>(HipA, HipB, HipC, M, N, K);
    hipDeviceSynchronize();

    ASSERT_EQ(hipMemcpy(C.data(), HipC, N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

    hipFree(HipA);
    hipFree(HipB);
    hipFree(HipC);
}