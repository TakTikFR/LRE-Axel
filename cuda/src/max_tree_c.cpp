#include "max_tree_c.hpp"
#include "max_tree.cuh"
#include "point.hpp"
#include "vector2D.hpp"
#include <cuda_runtime.h>
#include <utility>

/**
 * @brief Create a maxtree using a non-sorting algorithm on GPU.
 *
 * @param f Starting image.
 * @return Vector2D<Point> Parent image.
 */
Vector2D<Point> maxtree(Vector2D<int> f) {
  std::pair<size_t, size_t> shape = f.shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  Vector2D<Point> parent(rows, cols);

  kernelMaxtree(parent, f); // Caller function of the Cuda kernel.
  return parent;
}