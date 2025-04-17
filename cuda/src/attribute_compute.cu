#include "vector2D.cuh"

__global__ void kernel_computeAttribute(Vector2D<int> parent,
                                        Vector2D<int> area)
{
    int rows = area.getRows();
    int cols = area.getCols();
    int size = rows * cols;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int point = y * cols + x;
        if (point >= 0 && point < size)
        {
            int child_point = point;
            int parent_point = parent[child_point];
            while (child_point != parent_point)
            {
                area[parent_point] += 1;
                child_point = parent_point;
                parent_point = parent[child_point];
            }
        }
    }
}

void kernelComputeAttribute(Vector2D<int> parent, Vector2D<int> area)
{
    int rows = area.getRows();
    int cols = area.getCols();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);

    kernel_computeAttribute<<<numBlocks, threadsPerBlock>>>(parent, area);

    cudaDeviceSynchronize();
}