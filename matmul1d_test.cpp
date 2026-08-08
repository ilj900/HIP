#include <gtest/gtest.h>

#include "matrix.h"
#include "matmul.h"

TEST(matmul, ReturnsSuccess)
{
    FMatrix A(4, 4);
    FMatrix B(4, 4, 1u);
    FMatrix C(4, 4, true);
    MatMul1D(A, B, C, 4);
    C.Print();
}

TEST(Sanity, MatGen)
{
    FMatrix A(10, 10);
    A.Print();
}