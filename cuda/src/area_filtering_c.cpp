#include <iostream>

#include "area_filtering.cuh"
#include "utils.hpp"
#include "vector2D.cuh"

/**
 * @brief Computute the area from a maxtree
 *
 * @param f Beginning image.
 * @param parent Max tree representation.
 * @param area Area matrix of the image.
 * @param filterValue Value where we start to apply the filter.
 * @return Vector2D<Point> Area image.
 */
__host__ Vector2D<int> areaFiltering(Vector2D<int>& f, Vector2D<int> parent, Vector2D<int> area, int filterValue)
{
    int rows = f.getRows();
    int cols = f.getCols();
    int size = rows * cols;

    int* res_data = (int*)malloc(sizeof(int) * size);
    Vector2D<int> filteredImage_result(rows, cols, res_data);

    int* d_f;
    int* d_parent;
    int* d_area;
    int* d_filteredImage;
    cudaMalloc(&d_f, size * sizeof(int));
    cudaMalloc(&d_parent, size * sizeof(int));
    cudaMalloc(&d_area, size * sizeof(int));
    cudaMalloc(&d_filteredImage, size * sizeof(int));

    cudaMemcpy(d_f, f.getData(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, parent.getData(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_area, area.getData(), size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_filteredImage, f.getData(), size * sizeof(int),
               cudaMemcpyHostToDevice);

    Vector2D<int> f_dev(rows, cols, d_f);
    Vector2D<int> parent_dev(rows, cols, d_parent);
    Vector2D<int> area_dev(rows, cols, d_area);
    Vector2D<int> filteredImage_dev(rows, cols, d_filteredImage);

    kernelAreaFiltering(f_dev, parent_dev, area_dev, filteredImage_dev, filterValue);

    cudaMemcpy(filteredImage_result.getData(), d_filteredImage, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_f);
    cudaFree(d_parent);
    cudaFree(d_area);
    cudaFree(d_filteredImage);

    return filteredImage_result;
}