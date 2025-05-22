#include <iostream>

#include "depth_image.cuh"
#include "utils.hpp"
#include "vector2D.cuh"

/**
 * @brief Create the depth image of from a parent matrix on GPU.
 *
 * @param parent Parent image.
 * @return Vector2D<Point> Parent image.
 */
__host__ Vector2D<int> depthImage(Vector2D<int>& parent)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int* res_data = (int*)malloc(sizeof(int) * size);
    Vector2D<int> depth_result(rows, cols, res_data);

    int* d_parent;
    int* d_depth;
    cudaMalloc(&d_parent, size * sizeof(int));
    cudaMalloc(&d_depth, size * sizeof(int));

    cudaMemcpy(d_parent, parent.getData(), size * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> init_depth(size, 0);
    cudaMemcpy(d_depth, init_depth.data(), size * sizeof(int),
               cudaMemcpyHostToDevice);

    Vector2D<int> parent_dev(rows, cols, d_parent);
    Vector2D<int> depth_dev(rows, cols, d_depth);

    kernelImageDepth(parent_dev, depth_dev);

    cudaMemcpy(depth_result.getData(), d_depth, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_parent);
    cudaFree(d_depth);

    return depth_result;
}