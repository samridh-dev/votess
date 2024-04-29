#include <cmath>
#include <libsycl.hpp>

template <typename T> 
inline T utils::square(const T x) {
  return x * x;
}

template <typename T> 
inline T utils::cube(const T x) {
  return x * x * x;
}

template <typename T> 
inline T utils::bmax(const T a, const T b) {
  const int cond = (a < b);
  return !cond * a + cond * b;
}

template <typename T> 
inline T utils::bmin(const T a, const T b) {
  const int cond = (a > b);
  return !cond * a + cond * b;
}

template <typename T> 
inline T utils::bmax(const T a, const T b, const T c) {
  const T ac = bmax(a, b);
  return bmax(ac, c);
}

template <typename T> 
inline T utils::bmin(const T a, const T b, const T c) {
  const T ac = bmin(a, b);
  return bmin(ac, c);
}

template <typename T>
inline T utils::bfmod(const T x, const T y) {
  return x - sycl::trunc(x/y) * y;
}

template <typename T>
inline void utils::swap(T& a, T& b) {
  const T tmp = a;
  a = b;
  b = tmp;
}
