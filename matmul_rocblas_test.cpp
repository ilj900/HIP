#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul2d.h"
#include "matmul_rocblas.h"
#include "utils.h"

namespace
{
std::filesystem::path BenchPath = "bench_data";
}

// M x K x N
constexpr std::array ValidationShapes =
{
    FShape{1, 1, 1},
    FShape{4, 4, 4},
    FShape(13, 17, 19),
    FShape(251, 251, 251),
    FShape{512, 4, 4},
    FShape{4, 4, 512},
    FShape{4, 512, 4},
    FShape(257, 256, 256),
    FShape(256, 257, 256),
    FShape(256, 256, 257),
    FShape{1, 256, 256},
    FShape{256, 256, 1},
    FShape{256, 1, 256},
    FShape{1, 1, 256},
    FShape{256, 1, 1},
    FShape{4096, 4096, 4096},
    FShape{10000, 10000, 10000},
};

constexpr std::array BenchShapes =
{
    FShape{256, 256, 256},
    FShape{1024, 1024, 1024},
    FShape{4096, 4096, 4096},
    FShape{10000, 10000, 10000},
    FShape{8192, 128, 8192},
    FShape{128, 8192, 128},
    FShape{1, 4096, 4096},
    FShape{8, 4096, 4096},
    FShape{32, 4096, 4096},
};

class MatmulRocBlasSanity : public ::testing::TestWithParam<FShape>
{};

class MatmulRocBlasBench : public ::testing::TestWithParam<FShape>
{};

std::string GetNameRocBlas(const FShape& Shape)
{
    return std::format("MatmulRocBlas_Shape_{}x{}x{}", Shape.M, Shape.N, Shape.K);
}

TEST_P(MatmulRocBlasSanity, Sanity)
{
    auto Shape = GetParam();
    auto& [M, N, K] = Shape;

    std::println("Sanity: {}", GetNameRocBlas(Shape));

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);
    FMatrix D = FMatrix::Zeros(M, N);
    MatMulRocblas(A, B, C);
    MatMul2D(A, B, D, {64, 4});
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    std::println();
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST_P(MatmulRocBlasBench, Bench)
{
    auto Shape = GetParam();
    auto& [M, N, K] = Shape;
    auto Name = GetNameRocBlas(Shape);

    std::println("Bench 2D: {}", Name);

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C = FMatrix::Zeros(M, N);

    uint32_t Runs = 50;
    std::vector<float> Values(Runs);
    for (int i = 0; i < Runs; ++i)
    {
        Values[i] = MatMulRocblas(A, B, C);
    }

    SaveTimings(Values, BenchPath / ("Bench2D" + Name + ".dat"));
    std::println("{}ms\n", Values.back());
}

INSTANTIATE_TEST_SUITE_P(Matmul, MatmulRocBlasSanity, ::testing::ValuesIn(ValidationShapes),
    [](const ::testing::TestParamInfo<FShape>& Info)
{
    const auto& S = Info.param;
    return std::format("Sanity_{}", GetNameRocBlas(S));
});

INSTANTIATE_TEST_SUITE_P(Matmul, MatmulRocBlasBench, ::testing::ValuesIn(BenchShapes),
    [](const ::testing::TestParamInfo<FShape>& Info)
{
    const auto& S = Info.param;
    return std::format("Bench_{}", GetNameRocBlas(S));
});
