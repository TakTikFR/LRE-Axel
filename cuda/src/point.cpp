#include "point.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <ostream>

//__host__ __device__ constexpr Point::Point() : x{-1}, y{-1} {}

// __device__ constexpr Point::Point(int x_, int y_) : x{x_}, y{y_} {}
/*
bool Point::operator==(const Point &p) const { return x == p.x && y == p.y; }

bool Point::operator!=(const Point &p) const { return x != p.x || y != p.y; }

bool Point::operator>(const Point &p) const {
  return std::sqrt(x * x + y * y) > std::sqrt(p.x * p.x + p.y * p.y);
}

bool Point::operator<(const Point &p) const {
  return std::sqrt(x * x + y * y) < std::sqrt(p.x * p.x + p.y * p.y);
}

bool Point::operator<=(const Point &p) const {
  return std::sqrt(x * x + y * y) <= std::sqrt(p.x * p.x + p.y * p.y);
}

bool Point::operator>=(const Point &p) const {
  return std::sqrt(x * x + y * y) >= std::sqrt(p.x * p.x + p.y * p.y);
}
*/

std::ostream &operator<<(std::ostream &os, const Point &p) {
  os << "(" << p.x << ", " << p.y << ")";
  return os;
}
