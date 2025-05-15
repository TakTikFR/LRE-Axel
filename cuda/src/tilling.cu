#include "max_tree.cuh"
#include "vector2D.cuh"

/**
 * @brief Find all the 4-neighbors that has been already visited (their parent
 * value is already set) and return them.
 *
 * @param p Pixel reference to find their neighbors.
 * @param parent Parent image.
 * @param neighbors List of all defined 4-neighbors (Or empty if no neighbors).
 * @return size_t Size of the list of neighbors;
 */
__device__ size_t undef_neighbors_t(int point, Vector2D<int>& parent,
                                    int* neighbors)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int x = point % cols;
    int y = point / cols;

    int dx[4] = { -1, 1, 0, 0 };
    int dy[4] = { 0, 0, -1, 1 };

    size_t count = 0;
    for (int i = 0; i < 4; ++i)
    {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows)
        {
            int np = ny * cols + nx;
            int neighbor = parent[np];
            if (neighbor != -1)
                neighbors[count++] = neighbor;
        }
    }

    return count;
}

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
    int size = rows * cols;

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
    int neighbors[4];
    size_t nb_size = undef_neighbors_t(p, parent, neighbors);
    for (size_t i = 0; i < nb_size; i++)
    {
        int neighbor = neighbors[i];
        connect(p, neighbor, f, parent);
    }
}

/**
 * @brief Cuda kernel function caller.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param f Starting image.
 */
void kernelTillingMaxtree(Vector2D<int> parent, Vector2D<int> f)
{
    int blockSize = 4;

    int rows = f.getRows();
    int cols = f.getCols();
    int size = rows * cols;

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