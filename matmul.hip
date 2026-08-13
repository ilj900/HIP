#include <hip/hip_runtime.h>

#include <vector>

#include "hip_utils.h"
#include "matmul.h"

__global__ void matmul1d(const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    uint I = blockIdx.x * blockDim.x + threadIdx.x;
    const uint Size = M * N;

    if (I >= Size)
    {
        return;
    }

    uint Row = I / N;
    uint Col = I % N;

    float Acc = 0;
    for (uint i = 0; i < K; ++i)
    {
        float L = A[Row * K + i];
        float R = B[Col + (N * i)];
        Acc += L * R;
    }
    C[I] = Acc;
}

float MatMul1D(const FMatrix &A, const FMatrix &B, FMatrix &C, uint32_t BlockSize)
{
    auto HipA = A.ToDevice();
    auto HipB = B.ToDevice();
    auto HipC = C.AllocateDevice();
    uint32_t M = C.M;
    uint32_t N = C.K;
    uint32_t K = A.K;

    FHipTimer Timer;
    Timer.Start();
    matmul1d<<<(C.Size() + BlockSize - 1) / BlockSize, BlockSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K);
    HipCheck(hipGetLastError());
    auto Ms = Timer.Stop();

    HipC.FromDevice(C.GetData(), C.SizeB());
    return Ms;
}