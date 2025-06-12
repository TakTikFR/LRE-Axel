#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "vector2D.cuh"

/**
 * @brief Print method for a Vector2D of int.
 *
 * @param image Image to be printed.
 */
void printVector2D(Vector2D<int>& image) 
{
    int rows = image.getRows();
    int cols = image.getCols();

    int maxValue = image[0];
    for (int i = 1; i < rows * cols; ++i)
        if (image[i] > maxValue)
            maxValue = image[i];

    int width = std::to_string(maxValue).length() + 1;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int point = i * cols + j;
            std::cout << std::setw(width) << image[point];
        }

        std::cout << '\n';
    }

    std::cout << '\n';
}

/**
 * @brief Convert index 1D index to 2D index.
 *
 * @param index Initial 1D index.
 * @param cols Number of columns.
 * @param x x element of the 2D index.
 * @param y y element of the 2D index.
 */
void indexToCoords(int index, int cols, int& x, int& y)
{
    y = index / cols;
    x = index % cols;
}

/**
 * @brief Create a graphviz text and png file.
 *
 * @param parent Parent image.
 * @param f Initial image.
 * @param filename Filename of the text and png graphviz file.
 */
void displayGraph(Vector2D<int>& parent, Vector2D<int>& f,
                  const std::string& filename)
{
    std::ofstream fout(filename + ".dot");
    fout << "digraph G {\n";

    int rows = parent.getRows();
    int cols = parent.getCols();
    int size = rows * cols;

    for (int i = 0; i < size; ++i)
    {
        int parent_index = parent[i];

        if (parent_index != -1 && parent_index != i)
        {
            int y1, x1, y2, x2;
            indexToCoords(i, cols, x1, y1);
            indexToCoords(parent_index, cols, x2, y2);

            int value_from = f[i];
            int value_to = f[parent_index];

            fout << "    \"(" << x1 << ", " << y1 << ") [" << value_from
                 << "]\""
                 << " -> "
                 << "\"(" << x2 << ", " << y2 << ") [" << value_to << "]\";\n";
        }
    }

    fout << "}\n";
    fout.close();

    std::string cmd = "dot -Tpng " + filename + ".dot -o " + filename + ".png";
    system(cmd.c_str());
}

/**
 * @brief Sort in ascending order the initial image f by index relative to their
 * value in f.
 *
 * @param f Initial image.
 * @return int* Return the sorting array.
 */
int* sortVector2D(const Vector2D<int>& f)
{
    int rows = f.getRows();
    int cols = f.getCols();
    int size = rows * cols;

    std::vector<std::tuple<int, int>> indexed_values;

    for (int i = 0; i < rows * cols; ++i)
    {
        int value = f[i];
        indexed_values.emplace_back(i, value);
    }

    std::sort(indexed_values.begin(), indexed_values.end(),
              [](const std::tuple<int, int>& a, const std::tuple<int, int>& b) {
                  return std::get<1>(a) < std::get<1>(b);
              });

    int* result = new int[size];

    int index = 0;
    for (const auto& coord : indexed_values)
    {
        result[index++] = std::get<0>(coord);
    }

    return result;
}

/**
 * @brief Getting a simple and compressed representation of the tree.
 *
 * @param parent Parent image.
 * @param f Starting image.
 */
Vector2D<int> canonize_tree(Vector2D<int> parent, Vector2D<int> f)
{
    int rows = f.getRows();
    int cols = f.getCols();
    int size = rows * cols;

    int* sortedVec = sortVector2D(f);
    for (int p = 0; p < size; p++)
    {
        int q = parent[p];
        if (q != -1 && parent[q] != -1 && f[parent[q]] == f[q])
            parent[p] = parent[q];
    }

    return parent;
}

/**
 * @brief Convert an image to its grayscale version.
 *
 * @param path Path of the image.
 * @param rows
 * @param cols
 * @return int* Return the pointer of the grayscale image
 */
int* loadGrayImageAsVector(const std::string& path, int& rows, int& cols)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(128, 128));
    if (image.empty())
    {
        std::cerr << "Image not found\n";
        return nullptr;
    }

    rows = image.rows;
    cols = image.cols;
    int* vector = new int[rows * cols];

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            vector[i * cols + j] = static_cast<int>(image.at<uchar>(i, j));

    return vector;
}

/**
 * @brief Convert a matrix to its grayscale image representation
 *
 * @param matrix Matrix to convert.
 * @param filename output filename
 * @return int* Return the pointer of the grayscale image
 */
void matrixToImage(const Vector2D<int>& matrix, const std::string& filename) {
    int rows = matrix.getRows();
    int cols = matrix.getCols();

    cv::Mat image(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int point = i * cols + j;
            int pixel = std::clamp(matrix[point], 0, 255);
            image.at<uchar>(i, j) = static_cast<uchar>(pixel);
        }
    }

    cv::imwrite(filename, image);
}