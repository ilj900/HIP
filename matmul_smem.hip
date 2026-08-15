#include <hip/hip_runtime.h>

#include <stdexcept>
#include <vector>

#include "hip_utils.h"
#include "matmul_smem.h"

template <uint TILE>
__global__ void matmul_smem(const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    uint Tx = threadIdx.x;
    uint Ty = threadIdx.y;

    uint Col = blockIdx.x * TILE + threadIdx.x;
    uint Row = blockIdx.y * TILE + threadIdx.y;

    float Acc = 0;
    uint NumTiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < NumTiles; ++t)
    {
        uint aCol = t * TILE + Tx;
        uint bRow = t * TILE + Ty;

        As[Ty][Tx] = (Row < M && aCol < K) ? A[Row * K + aCol] : 0.0f;
        Bs[Ty][Tx] = (bRow < K && Col < N) ? B[bRow * N + Col] : 0.0f;

        __syncthreads();

        for (uint k = 0; k < TILE; ++k)
        {
            Acc += As[Ty][k] * Bs[k][Tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < N)
    {
        C[Row * N + Col] = Acc;
    }
}

float MatMulSMEM(const FMatrix &A, const FMatrix &B, FMatrix &C, const FTile& TileSize)
{
    if (TileSize.X != TileSize.Y)
    {
        throw std::runtime_error("MatMulSMEM requires a square tile (X == Y)");
    }

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

    switch(TileSize.X)
    {
        case 8: matmul_smem<8><<<GridSize, GroupSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K); break;
        case 16: matmul_smem<16><<<GridSize, GroupSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K); break;
        case 32: matmul_smem<32><<<GridSize, GroupSize>>>(HipA.Data, HipB.Data, HipC.Data, M, N, K); break;
        default: throw std::runtime_error("MatMulSMEM: unsupported tile size (use 8/16/32)");
    }

    HipCheck(hipGetLastError());
    auto Ms = Timer.Stop();

    HipC.FromDevice(C.GetData(), C.SizeB());
    return Ms;
}