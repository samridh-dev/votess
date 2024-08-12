#ifndef HEAP_HPP
#define HEAP_HPP

#include <libsycl.hpp>

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

namespace heap {

template <typename Ti, typename Tf>
inline void swap(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

template <typename Ti, typename Tf>
inline void maxheapify(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

template <typename Ti, typename Tf>
inline void sort(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

namespace heap {

template <typename Ti, typename Tf>
inline void swap(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

template <typename Ti, typename Tf>
inline void maxheapify(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

template <typename Ti, typename Tf>
inline void sort(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap

namespace heap {

template <typename Ti, typename Tf>
inline void swap(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int a, const int b
);
 
template <typename Ti, typename Tf>
inline void maxheapify(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int k, int idx
);

template <typename Ti, typename Tf>
inline void sort(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int k
);

} // namespace heap

#include <heap.ipp>
#endif
