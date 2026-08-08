#include <print>
#include <random>

#include "matrix.h"

FMatrix::FMatrix(uint32_t MIn, uint32_t KIn, float Min, float Max, uint32_t Seed) : M (MIn), K (KIn), Data(M * K)
{
    std::mt19937 Engine{Seed};
    std::uniform_real_distribution Dist(Min, Max);

    for (auto & Entry : Data)
    {
        Entry = Dist(Engine);
    }
}

void FMatrix::Print()
{
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            std::print("{:10.3f}", Data[i * K + j]);
        }
        std::println();
    }
}

bool FMatrix::operator==(const FMatrix& Other) const
{
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < K; ++j)
        {
            uint32_t Index = i * K + j;
            if (Other.Data[Index] != Data[Index])
            {
                return false;
            }
        }
    }
    return true;
}

