#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "max_tree.cuh"
#include "vector2D.cuh"

/**
 * @brief Cuda kernel for the area computation.
 *
 * @param f Original image.
 * @param parent Parent matrix.
 * @param area Area representation.
 * @param filteredImage Image after the application of the filter.
 * @param filterVale Value where we start to apply the filter.
 * @return __global__
 */
__global__ void kernel_areaFiltering(Vector2D<int> f, Vector2D<int> parent, Vector2D<int> area, Vector2D<int> filteredImage, int filterValue) 
{
    int rows = area.getRows();
    int cols = area.getCols();
    int size = rows * cols;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int point = y * cols + x;
        int par = parent[point];
        if (area[point] > filterValue || (area[point] == 1 && area[par] > filterValue))
        {
            //printf("point: (%d, %d), area: %d, parent: (%d, %d), area: %d\n", point / cols, point % cols, area[point], parent[point] / cols, parent[point] % cols, area[parent[point]]);
            filteredImage[point] = 0;
        }
    }
}

/**
 * @brief Cuda kernel function caller.
 *
 * @param f Original image.
 * @param parent Parent image, also the representation of the maxtree.
 * @param area Area image, represent the computation of the maxtree area.
 * @param filteredImage Image to store the data after the application of the filter.
 * @param filterVale Value where we start to apply the filter.
 */
void kernelAreaFiltering(Vector2D<int> f, Vector2D<int> parent, Vector2D<int> area, Vector2D<int> filteredImage, int filterValue)
{
    int rows = area.getRows();
    int cols = area.getCols();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);

    kernel_areaFiltering<<<numBlocks, threadsPerBlock>>>(f, parent, area, filteredImage, filterValue);

    int* cpuData = new int[rows * cols];
    cudaMemcpy(cpuData, filteredImage.getData(), rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}