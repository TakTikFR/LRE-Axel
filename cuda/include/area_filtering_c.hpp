#pragma once

#include "vector2D.cuh"

Vector2D<int> areaFiltering(Vector2D<int>& f, Vector2D<int> parent, Vector2D<int> area, int filterValue);