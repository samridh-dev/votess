#ifndef KNN_HPP
#define KNN_HPP

#include <arguments.hpp>
#include <xyzset.hpp>
#include <heap.hpp>
#include <libsycl.hpp>

namespace knni {

///////////////////////////////////////////////////////////////////////////////
/// K Nearest Neighbor Algorithm
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
void compute(
  const T1 i,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<T1>& id,
  const std::vector<T1>& offset,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const struct args::knn& args
);

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
void compute(
  const T1 i,
  const device_accessor_read_t<T2>& xyzset,
  const size_t xyzsize,
  const device_accessor_read_t<T1>& id,
  const device_accessor_read_t<T1>& offset,
  const device_accessor_read_t<T2>& refset,
  const size_t refsize,
  device_accessor_readwrite_t<T1> heap_id,
  device_accessor_readwrite_t<T2> heap_pq,
  const struct args::knn& args
);


/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// end
///////////////////////////////////////////////////////////////////////////////

} // namespace knni
  
#include <knn.ipp>
#endif
