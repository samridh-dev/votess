#ifndef HEAP_HPP
#define HEAP_HPP

#include <libsycl.hpp>

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

namespace heap {

template <typename T1, typename T2>
inline void swap(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

template <typename T1, typename T2>
inline void maxheapify(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

template <typename T1, typename T2>
inline void sort(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

namespace heap {

template <typename T1, typename T2>
inline void swap(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

template <typename T1, typename T2>
inline void maxheapify(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

template <typename T1, typename T2>
inline void sort(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap
  
#include <heap.ipp>
#endif
