#include "max_tree_c.hpp"
#include "utils.hpp"
#include "vector2D.hpp"

int main() {
  Vector2D<int> image(3, 3, std::vector<int>{0, 1, 1, 0, 2, 1, 3, 2, 0});

  std::cout << "Premier print: \n";
  printVector2D(image);
  Vector2D<Point> parent = maxtree(image);
  std::cout << "Deuxieme2 print: \n";
  printVector2D(parent);

  return 0;
}