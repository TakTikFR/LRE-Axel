#ifndef POINT_HPP
#define POINT_HPP

#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>

class Point {
public:
  int x, y;

  __host__ __device__ Point();
  __host__ __device__ Point(int x, int y);

  __host__ __device__ bool operator==(const Point &p) const;
  __host__ __device__ bool operator!=(const Point &p) const;
  __host__ __device__ bool operator>(const Point &p) const;
  __host__ __device__ bool operator<(const Point &p) const;
  __host__ __device__ bool operator<=(const Point &p) const;
  __host__ __device__ bool operator>=(const Point &p) const;
  friend std::ostream &operator<<(std::ostream &os, const Point &P);
};

const inline Point NULL_POINT(-1, -1);

#endif // POINT_HPP