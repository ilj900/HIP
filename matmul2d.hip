#include <hip/hip_runtime.h>

#include <vector>

#include "hip_utils.h"
#include "matmul2d.h"

__global__ void matmul2d(const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    uint Col = blockIdx.x * blockDim.x + threadIdx.x;
    uint Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Row >= M || Col >= N)
    {
        return;
    }

    float Acc = 0;
    for (uint i = 0; i < K; ++i)
    {
        float L = A[Row * K + i];
        float R = B[Col + (N * i)];
        Acc += L * R;
    }
    C[Row * N + Col] = Acc;
}

float MatMul2D(const FMatrix &A, const FMatrix &B, FMatrix &C, const FTile& TileSize)
{
    auto HipA = A.ToDevice();
    auto HipB = B.ToDevice();
    auto HipC = C.AllocateDevice();
    uint32_t M = C.M;
    uint32_t N = C.K;
    uint32_t K = A.K;

    dim3 GroupSize = {TileSize.X, TileSize.Y};
    dim3 GridSize = {(N + GroupSize.x - 1) / GroupSize.x, (M + GroupSize.y - 1) / GroupSize.y};

    FHipTimer Timer;
    Timer.Start();

    matmul2d<<<GridSize, GroupSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K);
    HipCheck(hipGetLastError());
    auto Ms = Timer.Stop();

    HipC.FromDevice(C.GetData(), C.SizeB());
    return Ms;
}