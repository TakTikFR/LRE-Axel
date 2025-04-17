#ifndef TILLING_CUH
#define TILLING_CUH

#pragma once

#include "vector2D.cuh"

void kernelTillingMaxtree(Vector2D<int> parent, Vector2D<int> f);

#endif // TILLING_CUH