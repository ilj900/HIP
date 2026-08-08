#include <hip/hip_runtime.h>
#include "hip_utils.h"

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
        float R = B[Col + (N * i)];
        Acc += L * R;
    }
    C[I] = Acc;
}

void MatMul1D(const FMatrix &A, const FMatrix &B, FMatrix &C, uint32_t BlockSize)
{
    auto HipA = A.ToDevice();
    auto HipB = B.ToDevice();
    auto HipC = C.ToDevice();
    uint32_t M = C.M;
    uint32_t N = C.K;
    uint32_t K = A.K;

    matmul1d<<<(C.Size() + BlockSize - 1) / BlockSize, BlockSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K);
    hipDeviceSynchronize();

    HipC.FromDevice(C.data(), C.SizeB());
}