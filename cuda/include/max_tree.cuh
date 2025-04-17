#ifndef MAX_TREE_CUH
#define MAX_TREE_CUH

#pragma once

#include "cuda_runtime.h"
#include "vector2D.cuh"
__device__ void connect(int& a, int& b, const Vector2D<int>& f,
                        Vector2D<int>& parent);
void kernelMaxtree(Vector2D<int> parent, Vector2D<int> f);
__device__ std::pair<int, int> findLevelRoot(Vector2D<int>& parent, int x,
                                             const Vector2D<int>& f);

#endif // MAX_TREE_CUH