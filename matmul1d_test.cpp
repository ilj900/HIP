#include <gtest/gtest.h>
#include <print>
#include "matrix.h"
#include "matmul.h"
#include "utils.h"

TEST(matmul, ReturnsSuccess)
{
    FMatrix A(4, 4);
    FMatrix B(4, 4, 1u);
    FMatrix C(4, 4, true);
    MatMul1D(A, B, C, 4);
    C.Print();
    auto D = CPUMatMul(A, B);
    D.Print();
    std::println("AbsDiff: {}", C.AbsDiff(D));
    std::println("RelDiff: {}", C.RelDiff(D));
    ASSERT_LE(C.RelDiff(D), 1e-6);
}

TEST(Sanity, MatGen)
{
    FMatrix A(10, 10);
    A.Print();
}