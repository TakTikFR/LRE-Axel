#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "max_tree.cuh"
#include "vector2D.cuh"

/**
 * @brief Cuda kernel for the area computation.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param area Area image.
 * @return __global__
 */
__global__ void kernel_computeArea(Vector2D<int> parent, Vector2D<int> area)
{
    int rows = area.getRows();
    int cols = area.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int point = y * cols + x;
        int child_point = point;
        int parent_point = parent[child_point];
        while (child_point != parent_point)
        {
            atomicAdd(&area[parent_point], 1);
            child_point = parent_point;
            parent_point = parent[child_point];
        }
    }
}

__global__ void kernel_normalizeArea(Vector2D<int> f, Vector2D<int> parent, Vector2D<int> area)
{
    int rows = area.getRows();
    int cols = area.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int point = y * cols + x;
        int par = parent[point];
        if (area[point] == 1 && f[point] == f[par])
            area[point] = area[par];
    }
}

/**
 * @brief Cuda kernel function caller.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param area Area image, represent the computation of the maxtree area.
 */
void kernelComputeArea(Vector2D<int> f, Vector2D<int> parent, Vector2D<int> area)
{
    int rows = area.getRows();
    int cols = area.getCols();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);

    kernel_computeArea<<<numBlocks, threadsPerBlock>>>(parent, area);
    cudaDeviceSynchronize();

    kernel_normalizeArea<<<numBlocks, threadsPerBlock>>>(f, parent, area);
    cudaDeviceSynchronize();
}