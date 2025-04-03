#include "max_tree.cuh"
#include "point.hpp"
#include "vector2D.hpp"
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
size_t undef_neighbors(Point p, Vector2D<Point> parent, Point *neighbors) {
  std::vector<Point> directions = {Point(0, -1), Point(0, 1), Point(-1, 0),
                                   Point(1, 0)};

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
bool totalOrderOp(Point &p, Point &q, const Vector2D<int> f) {
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
std::pair<Point, Point> findPeakRoot(Vector2D<Point> &parent, Point x, int lvl,
                                     Vector2D<int> f) {
  Point q = parent[x];
  while (q != x && lvl < f[q]) {
    Point old = q;
    q = parent[q];
    x = old;
  }

  return std::make_pair(x, q);
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
std::pair<Point, Point> findLevelRoot(Vector2D<Point> &parent, Point x,
                                      Vector2D<int> f) {
  return findPeakRoot(parent, x, f[x], f);
}

/**
 * @brief Connect two separate trees.
 *
 * @param a Pixel associated to the first tree.
 * @param b Pixel associated to the second tree.
 * @param f Starting image.
 * @param parent Parent image.
 */
void connect(Point &a, Point &b, const Vector2D<int> &f,
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