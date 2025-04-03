#ifndef MAX_TREE_CUH
#define MAX_TREE_CUH

#pragma once

#include "point.hpp"
#include "vector2D.hpp"

__device__ Point atomicPointMax(Point *address, Point val);
__global__ void kernel_maxtree(Vector2D<Point> *parent, Vector2D<int> f);
void kernelMaxtree(Vector2D<Point> &parent, Vector2D<int> f);

#endif // MAX_TREE_CUH