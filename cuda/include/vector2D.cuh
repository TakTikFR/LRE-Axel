#ifndef VECTOR2D_HPP
#define VECTOR2D_HPP

#pragma once

#include <cassert>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"

template <typename T>
class Vector2D
{
private:
    T* data;
    int rows, cols;

public:
    __host__ Vector2D(int rows_, int cols_, T* data_)
        : data{ data_ }
        , rows{ rows_ }
        , cols{ cols_ }
    {}

    __host__ __device__ int getRows() const
    {
        return rows;
    }

    __host__ __device__ int getCols() const
    {
        return cols;
    }

    __host__ __device__ T* getData() const
    {
        return data;
    }

    __host__ __device__ T& operator[](const int& index)
    {
        return data[index];
    }

    __host__ __device__ const T& operator[](const int& index) const
    {
        return data[index];
    }
};

#endif // VECTOR2D_CUH