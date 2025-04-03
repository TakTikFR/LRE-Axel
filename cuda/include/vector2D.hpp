#ifndef VECTOR2D_HPP
#define VECTOR2D_HPP

#pragma once

#include "point.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

class Point;

template <typename T> class Vector2D {
private:
  T *data;
  size_t rows, cols;

public:
  __host__ __device__ Vector2D(size_t rows_, size_t cols_, const T &value = T())
      : rows{rows_}, cols{cols_} {

    cudaMallocManaged(&data, rows * cols * sizeof(T));
    for (size_t i = 0; i < rows * cols; ++i) {
      data[i] = value;
    }
  }

  __host__ __device__ Vector2D(size_t rows_, size_t cols_,
                               const std::vector<T> &data_)
      : rows{rows_}, cols{cols_} {

    if (data_.size() != rows * cols)
      throw std::invalid_argument("Invalid initial data size");

    cudaMallocManaged(&data, rows * cols * sizeof(T));

    std::memcpy(data, data_.data(), rows * cols * sizeof(T));

    cudaDeviceSynchronize();
  }

  __host__ __device__ ~Vector2D() { cudaFree(data); }

  __host__ __device__ T &operator[](const Point &p) {
    check_bounds(p);
    return data[p.y * cols + p.x];
  }

  __host__ __device__ const T &operator[](const Point &p) const {
    check_bounds(p);
    return data[p.y * cols + p.x];
  }

  __host__ __device__ std::pair<size_t, size_t> shape() const {
    return {rows, cols};
  }

private:
  __host__ __device__ void check_bounds(const Point &p) const {
    assert(p.x >= 0 && static_cast<size_t>(p.x) < cols && p.y >= 0 &&
           static_cast<size_t>(p.y) < rows);
  }
};

#endif // VECTOR2D_HPP