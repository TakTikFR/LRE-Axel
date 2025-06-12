#include "tilling_c.hpp"

#include <iostream>

#include "max_tree.cuh"
#include "tilling.cuh"
#include "utils.hpp"
#include "vector2D.cuh"

/**
 * @brief Create a maxtree using the tilling method on GPU.
 *
 * @param f Starting image.
 * @return Vector2D<Point> Parent image.
 */
Vector2D<int> tillingMaxtree(Vector2D<int>& f)
{
    int rows = f.getRows();
    int cols = f.getCols();
    int size = rows * cols;

    int* res_data = (int*)malloc(sizeof(int) * size);
    Vector2D<int> parent_result(rows, cols, res_data);

    int* d_f;
    int* d_parent;
    cudaMalloc(&d_f, size * sizeof(int));
    cudaMalloc(&d_parent, size * sizeof(int));

    cudaMemcpy(d_f, f.getData(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_parent, -1, size * sizeof(int));

    Vector2D<int> f_dev(rows, cols, d_f);
    Vector2D<int> parent_dev(rows, cols, d_parent);

    kernelTillingMaxtree(parent_dev, f_dev);

    cudaMemcpy(parent_result.getData(), d_parent, size * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_f);
    cudaFree(d_parent);

    return parent_result;
}