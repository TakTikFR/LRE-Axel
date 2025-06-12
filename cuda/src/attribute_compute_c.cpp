#include <iostream>

#include "attribute_compute.cuh"
#include "utils.hpp"
#include "vector2D.cuh"
#include <benchmark/benchmark.h>

/**
 * @brief Computute the area from a maxtree
 *
 * @param f Starting image.
 * @return Vector2D<Point> Area image.
 */
__host__ Vector2D<int> computeArea(Vector2D<int>& f, Vector2D<int>& parent)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int* res_data = (int*)malloc(sizeof(int) * size);
    Vector2D<int> area_result(rows, cols, res_data);

    int* d_f;
    int* d_parent;
    int* d_area;
    cudaMalloc(&d_f, size * sizeof(int));
    cudaMalloc(&d_parent, size * sizeof(int));
    cudaMalloc(&d_area, size * sizeof(int));

    cudaMemcpy(d_parent, parent.getData(), size * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> init_area(size, 1);
    cudaMemcpy(d_area, init_area.data(), size * sizeof(int),
               cudaMemcpyHostToDevice);

    Vector2D<int> f_dev(rows, cols, d_f);
    Vector2D<int> parent_dev(rows, cols, d_parent);
    Vector2D<int> area_dev(rows, cols, d_area);


    kernelComputeArea(f_dev, parent_dev, area_dev);

    cudaMemcpy(area_result.getData(), d_area, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_f);
    cudaFree(d_parent);
    cudaFree(d_area);

    return area_result;
}