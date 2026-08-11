#include "utils.h"

#include <fstream>
#include <cstdint>

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

void SaveTimings(const std::vector<float>& Tmings, const std::filesystem::path& Filename)
{
    std::ofstream Out(Filename, std::ios::binary | std::ios::trunc);
    if (!Out) {
        throw std::runtime_error("saveTimings: failed to open '" + Filename.string() + "' for writing");
    }

    static const char magic[4] = {'T', 'I', 'M', '1'};
    Out.write(magic, sizeof(magic));

    const uint32_t Count = static_cast<uint32_t>(Tmings.size());
    Out.write(reinterpret_cast<const char*>(&Count), sizeof(Count));

    if (Count > 0)
    {
        Out.write(reinterpret_cast<const char*>(Tmings.data()), static_cast<std::streamsize>(Count) * sizeof(float));
    }

    if (!Out) {
        throw std::runtime_error("saveTimings: write failed for '" + Filename.string() + "'");
    }
}