#ifndef DEPTH_IMAGE_CUH
#define DEPTH_IMAGE_CUH

#pragma once

#include "cuda_runtime.h"
#include "vector2D.cuh"
__device__ void kernelImageDepth(Vector2D<int> parent, Vector2D<int> depthImage);

#endif // DEPTH_IMAGE_CUH