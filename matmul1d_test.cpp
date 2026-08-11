#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul.h"
#include "utils.h"
#include <array>

std::filesystem::path BenchPath = "bench_data";

struct FShape
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

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

constexpr std::array BlockSizes1D =
{
    64u,
    128u,
    255u,
    256u,
    257u,
    1024u
};

class MatmulVerificator : public ::testing::TestWithParam<FShape>
{};

class Matmul1DBench : public ::testing::TestWithParam<std::tuple<FShape, uint32_t>>
{};

std::string GetName1D(const FShape& Shape, uint32_t BlockSize)
{
    return std::format("GroupSize_{}_{}x{}x{}", BlockSize, Shape.M, Shape.N, Shape.K);
}

TEST(matmul, ReturnsSuccess)
{
    FMatrix A(3, 3);
    FMatrix B(3, 3, 1u);
    FMatrix C(3, 3, true);
    MatMul1D(A, B, C, 4);
    A.PrintWolfram();
    std::println();
    B.PrintWolfram();
    std::println();
    C.Print();
    std::println();
    auto D = CPUMatMul(A, B);
    D.Print();
    std::println();
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST(Sanity, MatGen)
{
    FMatrix A(10, 10);
    A.Print();
}

TEST_P(Matmul1DBench, Bench1D)
{
    auto [Shape, BlockSize] = GetParam();

    auto& M = Shape.M;
    auto& N = Shape.N;
    auto& K = Shape.K;
    auto Name = GetName1D(Shape, BlockSize);

    std::println("Bench 1D: {}", Name);

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C(M, N, true);

    uint32_t Runs = 5;
    std::vector<float> Values(Runs);
    for (int i = 0; i < Runs; ++i)
    {
        Values[i] = MatMul1D(A, B, C, BlockSize);
    }

    SaveTimings(Values, BenchPath / ("Bench1D" + Name + ".dat"));
    std::println("{}ms\n", Values.back());
}

TEST_P(MatmulVerificator, TestCorrectness)
{
    auto Shape = GetParam();

    auto& M = Shape.M;
    auto& N = Shape.N;
    auto& K = Shape.K;

    std::println("Validation: {}", GetName1D(Shape, 256));

    FMatrix A(M, K);
    FMatrix B(K, N);
    FMatrix C(M, N, true);
    MatMul1D(A, B, C, 256);
    auto D = CPUMatMul(A, B);
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    std::println();
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

INSTANTIATE_TEST_SUITE_P(VerificationTest, MatmulVerificator, ::testing::ValuesIn(ValidationShapes),
    [](const ::testing::TestParamInfo<FShape>& Info)
{
        const auto& S = Info.param;
        return std::format("Validation_K{}", GetName1D(S, 256));
});

INSTANTIATE_TEST_SUITE_P(Bench1DTest, Matmul1DBench, ::testing::Combine(::testing::ValuesIn(BenchShapes), ::testing::ValuesIn(BlockSizes1D)),
    [](const ::testing::TestParamInfo<std::tuple<FShape, uint32_t>>& Info)
{
        const auto& S = std::get<0>(Info.param);
        const auto  B = std::get<1>(Info.param);
        return std::format("Bench1D_{}", GetName1D(S, B));
});
