#ifndef MAX_TREE_CUH
#define MAX_TREE_CUH

#pragma once

#include "cuda_runtime.h"
#include "vector2D.cuh"

__global__ void kernel_maxtree(Vector2D<int> parent, Vector2D<int> f);
void kernelMaxtree(Vector2D<int> parent, Vector2D<int> f);

#endif // MAX_TREE_CUH