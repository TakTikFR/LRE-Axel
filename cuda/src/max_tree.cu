#include "max_tree.cuh"
#include "max_tree_c.hpp"
#include "point.hpp"
#include "vector2D.hpp"
#include <cuda_runtime.h>
#include <utility>

/**
 * @brief Compare And Swap Cuda function for Point type.
 *        Check the word old located at the address address in global or shared
 *        memory, computes (old == compare ? val : old), and stores the result back to
 *        memory at the same address.
 *
 * @param address Object to compare.
 * @param compare Object to compare with.
 * @param val Object value to be set if equal.
 * @return __device__ Returns old.
 */
__device__ Point atomicCAS(Point *address, Point compare, Point val) {
  Point old = *address;
  *address = old == compare ? val : old;
  return old;
}

/**
 * @brief Maximum comparison Cuda function for Point Type.
 *        word old located at the address address in global or shared memory,
 *        computes the maximum of old and val, and stores the result back to memory at
 *        the same address. These three operations are performed in one atomic
 *        transaction. The function returns old.
 *
 * @param address Object to compare.
 * @param val Object to compare with.
 * @param f Starting image reference for the intensity.
 * @return __device__ Returns old.
 */
__device__ Point atomicMax(Point *address, Point val, Vector2D<int> f) {
  Point ret = *address;
  while (totalOrderOp(ret, val, f)) {
    Point old = ret;
    if ((ret = atomicCAS(address, old, val)) == old)
      break;
  }
  return ret;
}

/**
 * @brief Cuda kernel for the creation of the maxtree.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param f Starting image.
 * @return __global__
 */
__global__ void kernel_maxtree(Vector2D<Point> *parent, Vector2D<int> *f) {
  std::pair<size_t, size_t> shape = (*parent).shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows) {
    Point point(y, x);
    (*parent)[point] = point;
    Point neighbors[4];
    size_t size = undef_neighbors(point, *parent, neighbors);
    for (size_t i = 0; i < size; i++) {
      Point neighbor = neighbors[i];
      connect(point, neighbor, *f, *parent);
    }
  }
}

/**
 * @brief Cuda kernel function caller.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param f Starting image.
 */
void kernelMaxtree(Vector2D<Point> &parent, Vector2D<int> f) {
  std::pair<size_t, size_t> shape = f.shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  int N = rows * cols;

  int numBlocks = (N + 255) / 256;
  int threadsPerBlock = 256;

  kernel_maxtree<<<numBlocks, threadsPerBlock>>>(&parent, &f);
}
