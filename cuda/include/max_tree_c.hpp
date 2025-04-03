#ifndef MAX_TREE_C_HPP
#define MAX_TREE_C_HPP

#pragma once

#include "point.hpp"
#include "vector2D.hpp"
#include <utility>

__host__ __device__ size_t undef_neighbors(Point p, Vector2D<Point> parent,
                                           Point *neighbors);

__host__ __device__ bool totalOrderOp(Point &p, Point &q,
                                      const Vector2D<int> f);

__host__ __device__ std::pair<Point, Point>
findPeakRoot(Vector2D<Point> &parent, Point x, int lvl, Vector2D<int> f);

__host__ __device__ std::pair<Point, Point>
findLevelRoot(Vector2D<Point> &parent, Point x, Vector2D<int> f);

__host__ __device__ void connect(Point &a, Point &b, const Vector2D<int> &f,
                                 Vector2D<Point> &parent);

__host__ __device__ Vector2D<Point> maxtree(Vector2D<int> f);

#endif // MAX_TREE_C_HPP