#include "max_tree_c.hpp"
#include "tilling_c.hpp"
#include "utils.hpp"
#include "vector2D.cuh"
#include "attribute_compute_c.hpp"

#include <chrono>

/*
int main()
{
    /*
    int rows = 3;
    int cols = 3;
    int size = rows * cols;
    int* data = new int[9]{ 0, 1, 1, 0, 2, 1, 3, 2, 0 };
    Vector2D<int> f(rows, cols, data);
    */
    /*
    int* data = new int[400]{
        1,1,1,1,2,3,3,2,1,1, 2,2,2,1,1,0,0,1,2,3,
        1,2,3,4,4,5,5,3,2,1, 3,4,3,2,1,0,1,2,3,3,
        1,2,4,6,6,7,6,4,3,2, 4,5,4,3,2,1,1,2,3,2,
        2,3,5,7,7,7,6,5,4,3, 3,4,5,3,2,2,3,3,2,1,
        2,3,5,6,6,6,5,4,3,2, 2,3,4,2,1,1,2,1,1,1,
        1,2,4,5,5,5,4,3,3,2, 1,2,3,1,0,1,1,0,0,1,
        1,2,3,4,4,3,3,2,2,1, 0,1,2,1,1,2,1,1,0,1,
        1,1,2,3,2,2,2,1,1,1, 0,1,2,2,2,2,1,2,1,0,
        1,1,1,1,1,2,3,2,1,0, 1,2,3,3,2,1,1,2,1,1,
        0,0,1,2,2,3,4,3,2,1, 1,2,4,4,3,2,2,3,2,1,

        0,1,2,3,3,4,5,4,3,2, 1,2,3,3,3,3,2,3,2,1,
        1,1,2,3,4,5,5,4,3,2, 0,1,2,2,2,2,2,2,1,1,
        1,2,2,3,4,4,4,3,2,1, 0,1,1,1,1,1,1,1,1,0,
        1,2,2,2,3,3,3,2,1,0, 0,0,1,1,0,0,0,1,1,1,
        2,3,2,2,2,2,2,1,0,0, 0,0,0,0,0,0,1,2,2,2,
        3,4,3,2,1,1,0,0,0,0, 0,0,1,2,2,2,3,4,3,2,
        3,5,4,2,1,0,0,0,1,1, 1,1,2,3,4,4,5,6,5,3,
        2,4,4,3,1,0,0,1,2,3, 3,4,5,6,6,6,6,7,7,4,
        1,2,3,3,2,1,1,2,3,4, 5,6,6,7,7,7,6,6,5,3,
        0,1,2,2,2,1,2,3,4,5, 6,6,6,6,6,5,5,4,3,2
    };


    int* data = new int[100]{
        1,1,2,3,3,3,2,1,1,0,
        1,2,4,5,6,5,3,2,1,0,
        1,2,5,7,7,6,4,2,1,1,
        1,3,5,6,5,5,4,3,2,1,
        1,2,4,4,3,4,5,5,3,2,
        0,1,2,2,2,3,5,6,4,2,
        0,0,1,2,3,4,6,7,5,3,
        0,0,1,1,2,4,6,6,5,3,
        0,1,1,2,3,4,5,5,4,2,
        1,2,2,3,3,3,4,4,3,1
    };
    

    int* data = new int[256]{
        1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 1, 1, 0, 0, 0, 0, 1, 2, 3, 2, 3, 4,
        5, 4, 2, 3, 2, 1, 0, 1, 1, 0, 2, 3, 4, 3, 4, 5, 6, 5, 3, 4, 3, 2,
        1, 2, 1, 1, 2, 3, 4, 3, 4, 6, 7, 6, 3, 5, 4, 3, 2, 3, 2, 2,

        1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 1, 1, 0, 0, 0, 0, 1, 2, 3, 2, 3, 4,
        5, 4, 2, 3, 2, 1, 0, 1, 1, 0, 2, 3, 4, 3, 4, 5, 6, 5, 3, 4, 3, 2,
        1, 2, 1, 1, 2, 3, 4, 3, 4, 6, 7, 6, 3, 5, 4, 3, 2, 3, 2, 2,

        0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6,
        7, 6, 5, 4, 3, 2, 1, 1, 1, 0, 2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3,
        2, 2, 1, 1, 2, 3, 4, 5, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1,

        0, 0, 1, 2, 2, 3, 3, 4, 3, 3, 2, 2, 1, 1, 1, 0, 1, 1, 2, 3, 3, 4,
        4, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 2, 3, 4, 4, 5, 5, 6, 5, 5, 4, 4,
        3, 3, 3, 2, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1
    }
    */

    /*
    int rows;
    int cols;
    std::string path = "/home/axel/Documents/LRE/LRE-Axel/cuda/images/1080p/1080p_2.png";
    int *data = loadGrayImageAsVector(path, rows, cols);

    Vector2D<int> f(rows, cols, data);

    auto start = std::chrono::high_resolution_clock::now();

    Vector2D<int> parent = tillingMaxtree(f);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << path << " : " << "Temps d'exécution : " << duration.count() << " secondes" << std::endl;
    */

    /*
    Vector2D<int> canonized_parent = canonize_tree(parent, f);
    printVector2D(canonized_parent);
    displayGraph(canonized_parent, f, "graph_canonized_ref");

    return 0;
    */
//}
/*
int main() {
    for (int i = 1; i <= 7; ++i) {
        int rows;
        int cols;

        std::ostringstream oss;
        oss << "/home/axel/Documents/LRE/LRE-Axel/cuda/images/144p/144p_" << i << ".png";
        std::string path = oss.str();

        int *data = loadGrayImageAsVector(path, rows, cols);
        Vector2D<int> f(rows, cols, data);

        auto start = std::chrono::high_resolution_clock::now();

        Vector2D<int> parent = tillingMaxtree(f);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << path << " : Temps d'exécution : " << duration.count() << " secondes" << std::endl;
    }

    return 0;
}*/

int main() {
    int* data = new int[256]{
        1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 1, 1, 0, 0, 0, 0, 1, 2, 3, 2, 3, 4,
        5, 4, 2, 3, 2, 1, 0, 1, 1, 0, 2, 3, 4, 3, 4, 5, 6, 5, 3, 4, 3, 2,
        1, 2, 1, 1, 2, 3, 4, 3, 4, 6, 7, 6, 3, 5, 4, 3, 2, 3, 2, 2,

        1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 1, 1, 0, 0, 0, 0, 1, 2, 3, 2, 3, 4,
        5, 4, 2, 3, 2, 1, 0, 1, 1, 0, 2, 3, 4, 3, 4, 5, 6, 5, 3, 4, 3, 2,
        1, 2, 1, 1, 2, 3, 4, 3, 4, 6, 7, 6, 3, 5, 4, 3, 2, 3, 2, 2,

        0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6,
        7, 6, 5, 4, 3, 2, 1, 1, 1, 0, 2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3,
        2, 2, 1, 1, 2, 3, 4, 5, 7, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1,

        0, 0, 1, 2, 2, 3, 3, 4, 3, 3, 2, 2, 1, 1, 1, 0, 1, 1, 2, 3, 3, 4,
        4, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 2, 3, 4, 4, 5, 5, 6, 5, 5, 4, 4,
        3, 3, 3, 2, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1
    };
    
    int rows = 16;
    int cols = 16;

    Vector2D<int> f(rows, cols, data);

    Vector2D<int> parent = tillingMaxtree(f);
    //displayGraph(parent, f, "graph_canonized_ref");
    Vector2D<int> area = computeArea(parent);
    printVector2D(area);

    return 0;
}