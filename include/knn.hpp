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

template <typename Ti, typename Tf>
void compute(
  const Ti i, const Ti index,
  const std::vector<std::array<Tf,3>>& xyzset,
  const Ti xyzsize,
  const std::vector<Ti>& id,
  const std::vector<Ti>& offset,
  const std::vector<std::array<Tf,3>>& refset,
  const Ti refsize,
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
  const struct args::knn& args
);

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
void compute(
  const Ti i, const Ti index,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Ti>& id,
  const device_accessor_read_t<Ti>& offset,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const struct args::knn& args
);

/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
void compute(
  const Ti i, const Ti index,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Ti>& id,
  const device_accessor_read_t<Ti>& offset,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq, const Ti hoffs,
  const struct args::knn& args
);

///////////////////////////////////////////////////////////////////////////////
/// end
///////////////////////////////////////////////////////////////////////////////

} // namespace knni
  
#include <knn.ipp>
#endif
