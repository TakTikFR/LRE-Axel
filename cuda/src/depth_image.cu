#include "vector2D.cuh"
#include <stdio.h>

__device__ int maxDepth = 0;

/**
 * @brief Kernel that create a depth image from a maxtree.
 * 
 * @param parent Parent image.
 * @param depthImage Depth image.
 * @return __global__ 
 */
__global__ void kernel_findDepth(Vector2D<int> parent, Vector2D<int> depthImage) {
    int rows = parent.getRows();
    int cols = parent.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int point = y * cols + x;
        int old = point;
        int par = parent[point];

        while (par != -1 && par != old) {
            old = par;
            par = parent[par];
            depthImage[point]++;
        }

        atomicMax(&maxDepth, depthImage[point]);
    }
}

/**
 * @brief Normalization kernel to have a normalize matrix to ease the visualization.
 * 
 * @param depthImage Depth image to normalize.
 * @return __global__ 
 */
__global__ void kernel_normalizeDepth(Vector2D<int> depthImage) {
    int rows = depthImage.getRows();
    int cols = depthImage.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int point = y * cols + x;
        int coeff = 255 / maxDepth;
        depthImage[point] *= coeff;
    }
}
/**
 * @brief Cuda kernel for the creation of a depth image.
 * 
 * @param parent Maxtree.
 * @param depthImage Depth image.
 */
void kernelImageDepth(Vector2D<int> parent, Vector2D<int> depthImage)
{
    int rows = parent.getRows();
    int cols = parent.getCols();

    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((cols + blockSize - 1) / blockSize,
                 (rows + blockSize - 1) / blockSize);

    kernel_findDepth<<<gridDim, blockDim>>>(parent, depthImage);
    cudaDeviceSynchronize();

    kernel_normalizeDepth<<<gridDim, blockDim>>>(depthImage);
    cudaDeviceSynchronize();
}