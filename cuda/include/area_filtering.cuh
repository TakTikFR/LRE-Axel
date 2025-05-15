#pragma once

#include "vector2D.cuh"

void kernelAreaFiltering(Vector2D<int> f, Vector2D<int> parent, Vector2D<int> area, Vector2D<int> filteredImage, int filterValue);