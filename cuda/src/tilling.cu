#include "max_tree.cuh"
#include "vector2D.cuh"
#include "utils.hpp"

/**
 * @brief Cuda kernel to connect all the sub-trees by linking all the boundaries
 * of a tree to its neighbor's boundaries.
 *
 * @param parent Parent image / Maxtree representation
 * @param block_size Size of cuda kernel block.
 * @return __global__
 */
__global__ void kernel_connectBoundaries(Vector2D<int> f, Vector2D<int> parent,
                                         int block_size)
{
    int rows = parent.getRows();
    int cols = parent.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int p = y * cols + x;

    if (x % block_size == 0 && p - 1 >= 0)
    {
        int np = p - 1;
        connect(p, np, f, parent);
    }

    if (y % block_size == 0 && p - cols >= 0)
    {
        int np = p - cols;
        connect(p, np, f, parent);
    }
}

/**
 * @brief Cuda kernel to canonicalized the tree.
 *
 * @param f Starting image.
 * @param parent Parent image.
 * @return __global__
 */
__global__ void kernel_flatten_t(Vector2D<int> f, Vector2D<int> parent)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int p = y * cols + x;

    if (p >= size)
        return;

    int q = parent[p];
    parent[p] = findLevelRoot(parent, q, f).first;
}

/**
 * @brief Cuda kernel to create a maxtree by using the tilling method
 *
 * @param f
 * @param parent
 * @param block_size
 * @return __global__
 */
__global__ void kernel_tillingMaxtree(Vector2D<int> f, Vector2D<int> parent,
                                      int block_size)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int p = y * cols + x;
    if (p >= size)
        return;

    parent[p] = p;
    
    if (x + 1 < cols)
        connect(p, p + 1, f, parent);
    if (y + 1 < rows)
        connect(p, p + cols, f, parent);
}

/**
 * @brief Cuda kernel function caller.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param f Starting image.
 */
void kernelTillingMaxtree(Vector2D<int> parent, Vector2D<int> f)
{
    int blockSize = 16;
    int rows = f.getRows();
    int cols = f.getCols();

    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((cols + blockSize - 1) / blockSize,
                 (rows + blockSize - 1) / blockSize);

    kernel_tillingMaxtree<<<gridDim, blockDim>>>(f, parent, blockSize);
    cudaDeviceSynchronize();

    kernel_flatten_t<<<gridDim, blockDim>>>(f, parent);
    cudaDeviceSynchronize();

    kernel_connectBoundaries<<<gridDim, blockDim>>>(f, parent, blockSize);
    cudaDeviceSynchronize();

    kernel_flatten_t<<<gridDim, blockDim>>>(f, parent);
    cudaDeviceSynchronize();
}