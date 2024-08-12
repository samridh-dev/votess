#ifndef CC_HPP
#define CC_HPP

#include <arguments.hpp>
#include <xyzset.hpp>

#include <status.hpp>
#include <planes.hpp>
#include <sradius.hpp>
#include <boundary.hpp>

#include <libsycl.hpp>

namespace cci {

///////////////////////////////////////////////////////////////////////////////
/// Convex Cell Direct Neighbors Algorithm
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */
template <typename Ti, typename Tf, typename Tu>
void compute(
  const Ti i, const Ti index,
  std::vector<cc::state>& states,
  Tf* P, Tu* T, Tu* dR,
  std::vector<Ti>& knn,
  std::vector<Ti>& dknn,
  const std::vector<std::array<Tf,3>>& xyzset,
  const Ti xyzsize,
  const std::vector<std::array<Tf,3>>& refset,
  const Ti refsize,
  const struct args::cc& args
);

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf, typename Tu>
void compute(
  const Ti i, const Ti index,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<Tf>& P,
  const device_accessor_readwrite_t<Tu>& T,
  const device_accessor_readwrite_t<Tu>& dR,
  const device_accessor_readwrite_t<Ti>& knn,
  const device_accessor_readwrite_t<Ti>& dknn,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const struct args::cc& args
);

/* ------------------------------------------------------------------------- */


template <typename Ti, typename Tf, typename Tu>
void compute(
  const Ti i, const Ti index,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<Tf>& P,
  const device_accessor_readwrite_t<Tu>& T,
  const device_accessor_readwrite_t<Tu>& dR,
  const device_accessor_readwrite_t<Ti>& knn, const Ti koffs,
  const device_accessor_readwrite_t<Ti>& dknn,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const struct args::cc& args
);

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////

} // namespace dnni
  
#include <cc.ipp>
#endif // CC_HPP
