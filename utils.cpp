#include "utils.h"

FMatrix CPUMatMul(const FMatrix& A, const FMatrix& B)
{
    if (A.K != B.M)
    {
        throw std::runtime_error("Wrong matrix shape.");
    }

    uint32_t M = A.M;
    uint32_t N = B.K;
    uint32_t K = A.K;

    FMatrix C(M, N, true);

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t  j = 0; j < N; ++j)
        {
            float Acc = 0.f;

            for (int k = 0; k < K; ++k)
            {
                Acc += A.Data[i * K + k] * B.Data[k * N + j];
            }

            C.Data[i * N + j] = Acc;
        }
    }

    return C;
}