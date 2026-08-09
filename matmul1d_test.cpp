#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul.h"
#include "utils.h"
#include <array>

struct Shape
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

// M x K x N
constexpr std::array Shapes =
{
    Shape{1, 1, 1},
    Shape{4, 4, 4},
    Shape(13, 17, 19),
    Shape(251, 251, 251),
    Shape{512, 4, 4},
    Shape{4, 4, 512},
    Shape{4, 512, 4},
    Shape(257, 256, 256),
    Shape(256, 257, 256),
    Shape(256, 256, 257),
    Shape{1, 256, 256},
    Shape{256, 256, 1},
    Shape{256, 1, 256},
    Shape{1, 1, 256},
    Shape{256, 1, 1},
};



class BasicMatmulVerificator : public ::testing::TestWithParam<Shape>
{};

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

TEST_P(BasicMatmulVerificator, TestCorrectness)
{
    auto Shape = GetParam();

    auto& M = Shape.M;
    auto& N = Shape.N;
    auto& K = Shape.K;

    std::println("M{}_N{}_K{}", M, N, K);

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

INSTANTIATE_TEST_SUITE_P(BasicTest, BasicMatmulVerificator, ::testing::ValuesIn(Shapes),
    [](const ::testing::TestParamInfo<Shape>& Info)
{
        const auto& S = Info.param;
        return std::format("M{}_N{}_K{}", S.M, S.N, S.K);
});
