#ifndef DNN_HPP
#define DNN_HPP

#include <arguments.hpp>
#include <xyzset.hpp>

#include <status.hpp>
#include <planes.hpp>
#include <sradius.hpp>
#include <boundary.hpp>

#include <libsycl.hpp>

namespace dnni {

///////////////////////////////////////////////////////////////////////////////
/// Convex Cell Direct Neighbors Algorithm
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

template <typename T1, typename T2, typename T3>
void compute(
  const T1 i,
  std::vector<cc::state>& states,
  T2* P,
  T3* T,
  T3* dR,
  std::vector<T1>& knn,
  std::vector<T1>& dknn,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  const struct args::cc& args
);

template <typename T1, typename T2, typename T3>
void compute(
  const T1 i,
  const T1 index,
  std::vector<cc::state>& states,
  T2* P,
  T3* T,
  T3* dR,
  std::vector<T1>& knn,
  std::vector<T1>& dknn,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  const struct args::cc& args
);

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

template <typename T1, typename T2, typename T3>
void compute(
  const T1 i,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<T2>& P,
  const device_accessor_readwrite_t<T3>& T,
  const device_accessor_readwrite_t<T3>& dR,
  const device_accessor_readwrite_t<T1>& knn,
  const device_accessor_readwrite_t<T1>& dknn,
  const device_accessor_read_t<T2>& xyzset,
  const size_t xyzsize,
  const device_accessor_read_t<T2>& refset,
  const size_t refsize,
  const struct args::cc& args
);

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////

} // namespace dnni
  
#include <dnn.ipp>

#endif
