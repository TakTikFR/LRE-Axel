#include "max_tree.cuh"
#include "max_tree_c.hpp"
#include "point.hpp"
#include "vector2D.hpp"
#include <cuda_runtime.h>
#include <utility>

/**
 * @brief Find all the 4-neighbors that has been already visited (their parent
 * value is already set) and return them.
 *
 * @param p Pixel reference to find their neighbors.
 * @param parent Parent image.
 * @param neighbors List of all defined 4-neighbors (Or empty if no neighbors).
 * @return size_t Size of the list of neighbors;
 */
__device__ size_t undef_neighbors(Point p, Vector2D<Point> parent,
                                  Point *neighbors) {
  Point directions[4] = {Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)};

  std::pair<size_t, size_t> shape = parent.shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  size_t count = 0;
  for (Point direction : directions) {
    Point newPoint = Point(p.x + direction.x, p.y + direction.y);
    if (0 <= newPoint.x && newPoint.x < static_cast<int>(cols) &&
        0 <= newPoint.y && newPoint.y < static_cast<int>(rows)) {
      Point neighbor = parent[newPoint];
      if (neighbor != NULL_POINT)
        neighbors[count] = neighbor;
      count++;
    }
  }

  return count;
}

/**
 * @brief Implementation of a total order between pixels.
 *
 * @param p First pixel coordinates.
 * @param q Second pixel coordinates.
 * @param f Starting image
 * @return true If the first element is greater than the second element compared
 * to the total order properties.
 * @return false If the second element is greater than the first element
 * compared to the total order properties.
 */
__device__ bool totalOrderOp(Point &p, Point &q, const Vector2D<int> f) {
  if (f[p] < f[q])
    return true;
  else if (f[p] == f[q])
    return p > q;

  return false;
}

/**
 * @brief Find the peak root of a given node in the tree.
 *
 * @param parent Parent image.
 * @param x The starting node.
 * @param lvl Reference intensity level.
 * @param f Starting image.
 * @return std::pair<Point, Point> Pair containing the peak root and its parent
 * (Null Point if no parent exists).
 */
__device__ std::pair<Point, Point>
findPeakRoot(Vector2D<Point> &parent, Point x, int lvl, Vector2D<int> f) {
  Point q = parent[x];
  while (q != x && lvl < f[q]) {
    Point old = q;
    q = parent[q];
    x = old;
  }

  return {x, q};
}

/**
 * @brief Find the level root of a given node in the tree (The root with the
 * same intensity as the point X).
 *
 * @param parent Parent image.
 * @param x The starting node.
 * @param f Starting image.
 * @return std::pair<Point, Point> Pair containing the peak root and its parent
 * (Null Point if no parent exists).
 */
__device__ std::pair<Point, Point> findLevelRoot(Vector2D<Point> &parent,
                                                 Point x, Vector2D<int> f) {
  return findPeakRoot(parent, x, f[x], f);
}

// rajouter host device.
/**
 * @brief Connect two separate trees.
 *
 * @param a Pixel associated to the first tree.
 * @param b Pixel associated to the second tree.
 * @param f Starting image.
 * @param parent Parent image.
 */
__device__ void connect(Point &a, Point &b, const Vector2D<int> &f,
                        Vector2D<Point> &parent) {
  while (b != NULL_POINT) {
    if (f[b] < f[a])
      std::swap(a, b);

    auto flr = findLevelRoot(parent, a, f);
    auto fpr = findPeakRoot(parent, b, f[a], f);
    a = flr.first;
    b = fpr.first;

    if (totalOrderOp(b, a, f))
      std::swap(a, b);

    if (a == b)
      ;
    return;

    b = atomicPointMax(&parent[b], a);
  }
}

/**
 * @brief Compare And Swap Cuda function for Point type.
 *        Check the word old located at the address address in global or shared
 *        memory, computes (old == compare ? val : old), and stores the result
 * back to memory at the same address.
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

// Cast en int les points (de porc)

/**
 * @brief Maximum comparison Cuda function for Point Type.
 *        word old located at the address address in global or shared memory,
 *        computes the maximum of old and val, and stores the result back to
 * memory at the same address. These three operations are performed in one
 * atomic transaction. The function returns old.
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

// Gerer directement avec les indexs plutot que struct point.

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