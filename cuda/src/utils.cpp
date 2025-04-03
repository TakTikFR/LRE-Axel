#include "point.hpp"
#include "vector2D.hpp"
#include <iostream>

/**
 * @brief Print method for a Vector2D of int.
 *
 * @param image Image to be printed.
 */
void printVector2D(Vector2D<int> image) {
  std::pair<size_t, size_t> shape = image.shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      Point point(j, i);
      std::cout << image[point] << " ";
    }

    std::cout << '\n';
  }

  std::cout << '\n';
}

/**
 * @brief Print method for Vector2D of Point.
 *
 * @param image Image to be printed.
 */
void printVector2D(Vector2D<Point> image) {
  std::pair<size_t, size_t> shape = image.shape();
  size_t rows = shape.first;
  size_t cols = shape.second;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      Point point(i, j);
      std::cout << image[point] << " ";
    }

    std::cout << '\n';
  }

  std::cout << '\n';
}