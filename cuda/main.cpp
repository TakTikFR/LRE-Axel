#include "max_tree_c.hpp"
#include "utils.hpp"
#include "vector2D.cuh"

int main()
{
    int rows = 3;
    int cols = 3;
    int size = rows * cols;

    int* data = new int[9]{ 0, 1, 1, 0, 2, 1, 3, 2, 0 };
    Vector2D<int> f(rows, cols, data);

    std::cout << "Premier print: \n";
    printVector2D(f);
    Vector2D<int> parent = maxtree(f);

    std::cout << "Deuxieme print: \n";
    printVector2D(parent);
    displayGraph(parent, f, "graph");

    Vector2D<int> canonized_parent = canonize_tree(parent, f);
    printVector2D(canonized_parent);
    displayGraph(canonized_parent, f, "graph_canonized");

    return 0;
}