#include <hip/hip_runtime.h>

#include <map>
#include <stdexcept>
#include <vector>

#include "hip_utils.h"
#include "matmul_blocktile_1d.h"

template <uint BM, uint BN, uint BK, uint TM>
__global__ void matmul_blocktile_1d(const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    uint BlockRow = blockIdx.y;
    uint BlockCol = blockIdx.x;

    uint ThreadRow = threadIdx.x / BN;
    uint ThreadCol = threadIdx.x % BN;

    uint NumThreads = BM * BN / TM;
    uint NumOfTiles = (K + BK - 1) / BK;

    uint InnerRowA = threadIdx.x / BK;
    uint InnerColA = threadIdx.x % BK;
    uint StrideA = NumThreads / BK;

    uint InnerRowB = threadIdx.x / BN;
    uint InnerColB = threadIdx.x % BN;
    uint StrideB = NumThreads / BN;

    float ThreadResults[TM] {0};

    for (uint t = 0; t < NumOfTiles; ++t)
    {
        for (uint o = 0; o < BM; o += StrideA)
        {
            uint aRow = BlockRow * BM + InnerRowA + o;
            uint aCol = t * BK + InnerColA;
            As[(InnerRowA + o) * BK + InnerColA] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;
        }

        for (uint o = 0; o < BK; o += StrideB)
        {
            uint bRow = t * BK + InnerRowB + o;
            uint bCol = BlockCol * BN + InnerColB;
            Bs[(InnerRowB + o) * BN + InnerColB] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.f;
        }

        __syncthreads();

        for (uint dot = 0; dot < BK; ++dot)
        {
            float bTmp = Bs[dot * BN + ThreadCol];
            for (uint r = 0; r < TM; ++r)
            {
                ThreadResults[r] += As[(ThreadRow * TM + r) * BK + dot] * bTmp;
            }
        }

        __syncthreads();
    }

    uint cCol = BlockCol * BN + ThreadCol;
    for (uint r = 0; r < TM; ++r)
    {
        uint cRow = BlockRow * BM + ThreadRow * TM + r;
        if (cRow < M && cCol < N)
        {
            C[cRow * N + cCol] = ThreadResults[r];
        }
    }
}

using FLauncher = void (*)(dim3 grid, dim3 block, const float* A, const float* B, float* C, uint M, uint N, uint K);

template <uint BM, uint BN, uint BK, uint TM>
void LaunchBlockTile1D(dim3 grid, dim3 block, const float* A, const float* B, float* C, uint M, uint N, uint K)
{
    matmul_blocktile_1d<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
}

float MatMulBlockTile1D(const FMatrix &A, const FMatrix &B, FMatrix &C, uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM)
{
    if ((BM * BN) % TM != 0) throw std::runtime_error("TM must divide BM*BN");
    if ((BM * BN / TM) % BK != 0) throw std::runtime_error("As load stride must divide evenly");
    if ((BM * BN / TM) % BN != 0) throw std::runtime_error("Bs load stride must divide evenly");
    if (BM % TM != 0) throw std::runtime_error("TM must divide BM");
    uint32_t StrideA = (BM * BN / TM) / BK;
    uint32_t StrideB = (BM * BN / TM) / BN;
    if (BM % StrideA != 0) throw std::runtime_error("StrideA must divide BM (As load OOB)");
    if (BK % StrideB != 0) throw std::runtime_error("StrideB must divide BK (Bs load OOB)");

    auto HipA = A.ToDevice();
    auto HipB = B.ToDevice();
    auto HipC = C.AllocateDevice();
    uint32_t M = C.M;
    uint32_t N = C.K;
    uint32_t K = A.K;

    static const std::map<F1DBlockTilingConfig, FLauncher> LaunchTable =
    {
        {{64, 64, 8, 8}, &LaunchBlockTile1D< 64, 64, 8, 8>},
        {{64, 64, 8, 16}, &LaunchBlockTile1D<64, 64, 8, 16>},
        {{64, 64, 8, 32}, &LaunchBlockTile1D< 64, 64, 8, 32>},
    };

    uint32_t NumberOfThreads = BM * BN / TM;
    dim3 GroupSize = {NumberOfThreads, 1, 1};
    dim3 GridSize = {(N + BN - 1) / BN, (M + BM - 1) / BM};

    auto It = LaunchTable.find({BM, BN, BK, TM});
    if (It == LaunchTable.end())
        throw std::runtime_error("MatMulBlockTile1D: unsupported launch config");

    FHipTimer Timer;
    Timer.Start();
    It->second(GridSize, GroupSize, HipA.Data, HipB.Data, HipC.Data, M, N, K);

    HipCheck(hipGetLastError());
    auto Ms = Timer.Stop();

    HipC.FromDevice(C.GetData(), C.SizeB());
    return Ms;
}