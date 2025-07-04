#include "max_tree_c.hpp"
#include "tilling_c.hpp"
#include "utils.hpp"
#include "vector2D.cuh"
#include "attribute_compute_c.hpp"
#include "area_filtering_c.hpp"
#include "depth_image_c.hpp"
#include <iostream>
#include <string>

int* data1 = new int[256]{
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
int rows1 = 16;
int cols1 = 16;

int* data2 = new int[256]{
    36, 36, 73, 73, 109, 109, 146, 146, 73, 73, 36, 36, 0, 0, 0, 0,
    36, 73, 109, 73, 109, 146, 182, 146, 73, 109, 73, 36, 0, 36, 36, 0,
    73, 109, 146, 109, 146, 182, 218, 182, 109, 146, 109, 73, 36, 73, 36, 36,
    73, 109, 146, 109, 146, 218, 255, 218, 109, 182, 146, 109, 73, 109, 73, 73,

    36, 36, 73, 73, 109, 109, 146, 146, 73, 73, 36, 36, 0, 0, 0, 0,
    36, 73, 109, 73, 109, 146, 182, 146, 73, 109, 73, 36, 0, 36, 36, 0,
    73, 109, 146, 109, 146, 182, 218, 182, 109, 146, 109, 73, 36, 73, 36, 36,
    73, 109, 146, 109, 146, 218, 255, 218, 109, 182, 146, 109, 73, 109, 73, 73,

    0, 36, 73, 109, 146, 182, 218, 255, 218, 182, 146, 109, 73, 36, 0, 0,
    36, 73, 109, 146, 182, 218, 255, 218, 182, 146, 109, 73, 36, 36, 36, 0,
    73, 109, 146, 182, 218, 255, 255, 255, 218, 182, 146, 109, 73, 73, 36, 36,
    73, 109, 146, 182, 255, 255, 255, 218, 182, 146, 109, 73, 36, 36, 36, 36,

    0, 0, 36, 73, 73, 109, 109, 146, 109, 109, 73, 73, 36, 36, 36, 0,
    36, 36, 73, 109, 109, 146, 146, 182, 146, 146, 109, 109, 73, 73, 73, 36,
    36, 73, 109, 146, 146, 182, 182, 218, 182, 182, 146, 146, 109, 109, 109, 73,
    36, 73, 109, 146, 182, 218, 218, 218, 182, 146, 109, 73, 73, 36, 36, 36
    };

int rows2 = 16;
int cols2 = 16;

int* data3 = new int[256]{
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0, 155,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0, 155,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 155,   155, 155, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 155,   155, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45,  45,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45,  45,   45,   45, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

int rows3 = 16;
int cols3 = 16;

int* data4 = new int[256]{
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 253, 253,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

int rows4 = 16;
int cols4 = 16;

int* data5 = new int[1024]{
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 200, 200, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 200, 200, 200, 200, 200,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 200, 200, 200, 200, 200,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0, 155, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 155, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0, 155, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 155, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 155, 155, 155, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   155, 155, 155, 45, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 155, 155, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 155, 155, 45, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 45,  255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 45, 45,  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,  45, 45,  45,  45,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   45,  45,  45,  45,  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0, 255,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,  255, 255,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,  255, 255,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0, 255,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,  0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0
};

int rows5 = 32;
int cols5 = 32;

int* test1(std::string path) {
    int rows;
    int cols;
    int *data = loadGrayImageAsVector(path, rows, cols);
    return data;
}

void test2() {
    Vector2D<int> f(rows2, cols2, data2);
    matrixToImage(f, "originalImage.png");
    Vector2D<int> parent = maxtree(f);
    displayGraph(parent, f, "graph_canonized_ref");
    /*
    Vector2D<int> area = computeArea(f, parent);
    matrixToImage(area, "image_area.png");
    printVector2D(area);
    Vector2D<int> filteredImage = areaFiltering(f, parent, area, 10);
    Vector2D<int> depth_image = depthImage(f, parent);
    matrixToImage(depth_image, "iamgeDepth.png");
    printVector2D(f);
    printVector2D(filteredImage);
    matrixToImage(filteredImage, "filteredImage.png");
    */
}

void test3() {
    Vector2D<int> f(rows3, cols3, data3);
    matrixToImage(f, "OriginalImage.png");
    Vector2D<int> parent = tillingMaxtree(f);
    displayGraph(parent, f, "parent_test");
    Vector2D<int> depth = depthImage(f, parent);
    printVector2D(depth);
    matrixToImage(depth, "DepthImage.png");
}

#include "max_tree_c.hpp"
#include "tilling_c.hpp"
#include "utils.hpp"
#include "vector2D.cuh"
#include "attribute_compute_c.hpp"
#include "area_filtering_c.hpp"
#include "depth_image_c.hpp"
#include "benchmark_area.cuh"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    
    runBenchmarks();
    
    benchmark::Shutdown();
    return 0;
}