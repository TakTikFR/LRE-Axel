#ifndef MAX_TREE_C_HPP
#define MAX_TREE_C_HPP

#pragma once

#include "point.hpp"
#include "vector2D.hpp"
#include <utility>

__host__ __device__ Vector2D<Point> maxtree(Vector2D<int> f);

#endif // MAX_TREE_C_HPP