#ifndef UTILS_HPP
#define UTILS_HPP

#pragma once

#include <cstdio>
#include <iostream>
#include <ostream>

#include "vector2D.cuh"

void printVector2D(Vector2D<int>& image);
void displayGraph(Vector2D<int>& parent, Vector2D<int>& img,
                  const std::string& filename);
Vector2D<int> canonize_tree(Vector2D<int> parent, Vector2D<int> f);
int* loadGrayImageAsVector(const std::string& path, int& rows, int& cols);
void matrix2image(const Vector2D<int> image, const std::string& filename);
void matrixToImage(const Vector2D<int>& matrix, const std::string& filename);

#endif // UTILS_HPP