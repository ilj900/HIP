#pragma once

#include "matrix.h"
#include "utils.h"

float MatMulBlockTile1D(const FMatrix &A, const FMatrix &B, FMatrix &C, uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM);