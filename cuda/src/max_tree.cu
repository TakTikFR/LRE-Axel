#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>

#include "max_tree.cuh"
#include "max_tree_c.hpp"
#include "vector2D.cuh"
#include "utils.hpp"

/**
 * @brief Find all the 4-neighbors that has been already visited (their parent
 * value is already set) and return them.
 *
 * @param p Pixel reference to find their neighbors.
 * @param parent Parent image.
 * @param neighbors List of all defined 4-neighbors (Or empty if no neighbors).
 * @return size_t Size of the list of neighbors;
 */
__device__ size_t undef_neighbors(int point, Vector2D<int>& parent,
                                  int* neighbors)
{
    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    int directions[4] = { /*left*/ -1,
                          /*Right*/ 1,
                          /*Upper*/ -cols,
                          /*Lower*/ +cols };

    size_t count = 0;
    for (int direction : directions)
    {
        int newPoint = point + direction;

        if (newPoint >= 0 && newPoint < size)
        {
            int neighbor = parent[newPoint];
            if (neighbor != -1) {
                neighbors[count++] = neighbor;
            }
        }
    }

    return count;
}

/**
 * @brief Implementation of a total order between pixels.
 *
 * @param p First pixel coordinates.
 * @param q Second pixel coordinates.
 * @param f Starting image
 * @return true If the first element is greater than the second element compared
 * to the total order properties.
 * @return false If the second element is greater than the first element
 * compared to the total order properties.
 */
__device__ bool totalOrderOp(int& p, int& q, const Vector2D<int>& f)
{
    int rows = f.getRows();
    int cols = f.getCols();

    if (f[p] < f[q]) {
        return true;
    } else if (f[p] == f[q]) {
        int py = p / cols;  // y
        int px = p % cols;  // x
        int qy = q / cols;
        int qx = q % cols;

        if (px > qx)
            return true;
        else if (px == qx && py > qy)
            return true;
        else
            return false;
    }

    return false;
}

/**
 * @brief Find the peak root of a given node in the tree.
 *
 * @param parent Parent image.
 * @param x The starting node.
 * @param lvl Reference intensity level.
 * @param f Starting image.
 * @return std::pair<Point, Point> Pair containing the peak root and its parent
 * (Null Point if no parent exists).
 */
__device__ std::pair<int, int> findPeakRoot(Vector2D<int>& parent, int x,
                                            int lvl, const Vector2D<int>& f)
{
    int q = parent[x];
    while (q != -1 && q != x && lvl <= f[q])
    {
        int old = q;
        q = parent[q];
        x = old;
    }

    return { x, q };
}

/**
 * @brief Find the level root of a given node in the tree (The root with the
 * same intensity as the point X).
 *
 * @param parent Parent image.
 * @param x The starting node.
 * @param f Starting image.
 * @return std::pair<Point, Point> Pair containing the peak root and its parent
 * (Null Point if no parent exists).
 */
__device__ std::pair<int, int> findLevelRoot(Vector2D<int>& parent, int x,
                                             const Vector2D<int>& f)
{
    return findPeakRoot(parent, x, f[x], f);
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// rajouter host device.
/**
 * @brief Connect two separate trees.
 *
 * @param a Pixel associated to the first tree.
 * @param b Pixel associated to the second tree.
 * @param f Starting image.
 * @param parent Parent image.
 */
__device__ void connect(int a, int b, const Vector2D<int>& f,
                        Vector2D<int>& parent)
{
    while (b != -1)
    {
        if (f[b] < f[a])
            swap(a, b);

        auto [newA, A] = findLevelRoot(parent, a, f);
        a = newA;
        auto [newB, B] = findPeakRoot(parent, b, f[a], f);
        b = newB;

        if (totalOrderOp(b, a, f))
        {
            swap(a, b);
            swap(A, B);
        }

        if (a == b)
            return;

        int old = atomicCAS(&parent[b], B, a);
        if (old == B)
            b = old;
    }
}

/**
 * @brief Cuda kernel to canonicalized the tree.
 *
 * @param f Starting image.
 * @param parent Parent image.
 * @return __global__
 */
__global__ void kernel_flatten(Vector2D<int> f, Vector2D<int> parent)
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
 * @brief Cuda kernel for the creation of the maxtree.
 *
 * @param parent Parent image, also the representation of the maxtree.
 * @param f Starting image.
 * @return __global__
 */
__global__ void kernel_maxtree(Vector2D<int> f, Vector2D<int> parent)
{
    int rows = parent.getRows();
    int cols = parent.getCols();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int p = y * cols + x;

    if (x < cols && y < rows)
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
void kernelMaxtree(Vector2D<int> parent, Vector2D<int> f)
{
    int rows = f.getRows();
    int cols = f.getCols();

    int blockSize = f.getRows();
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((cols + blockSize - 1) / blockSize,
                 (rows + blockSize - 1) / blockSize);

    kernel_maxtree<<<gridDim, blockDim>>>(f, parent);
    cudaDeviceSynchronize();

    kernel_flatten<<<gridDim, blockDim>>>(f, parent);
    cudaDeviceSynchronize();
}