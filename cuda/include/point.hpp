#ifndef POINT_HPP
#define POINT_HPP

#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>

struct Point {
  int x, y;

  __host__ __device__ auto operator<=>(const Point &) const = default;
  friend std::ostream &operator<<(std::ostream &os, const Point &P);
};

__device__ constexpr inline Point NULL_POINT = Point(-1, -1);

#endif // POINT_HPP