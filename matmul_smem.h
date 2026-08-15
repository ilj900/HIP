#pragma once

#include "matrix.h"
#include "utils.h"

float MatMulSMEM(const FMatrix &A, const FMatrix &B, FMatrix &C, const FTile& TileSize);