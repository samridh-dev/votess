#ifndef BOUNDARY_HPP
#define BOUNDARY_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

#include <libsycl.hpp>

namespace boundary {

///////////////////////////////////////////////////////////////////////////////
/// Internal 
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

struct vstatus {
  uint8_t byte;
  inline vstatus();
  inline uint8_t get(int b);
  inline void set_true(int b);
  inline uint8_t get_nshared_edges() const;
  inline uint8_t get_shared_position() const;
};

enum bstatus : unsigned char {
  success,
  unreachable,
  undefined = 0xff
};

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Compute Boundary function 
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

template<typename Ti, typename Tu>
inline bstatus compute(
  Tu* cycle, 
  const Ti dr_offs,
  const Ti dr_size,
  short int& head,
  Tu* R,
  const Ti r_offs,
  const Ti r_size
);

/* ------------------------------------------------------------------------- */
/// SYCL implementation
/* ------------------------------------------------------------------------- */

template<typename Ti, typename Tu>
inline bstatus compute(
  const sycl::local_accessor<Tu, 1>& cycle,
  const Ti dr_offs,
  const Ti dr_size,
  short int& head,
  const device_accessor_readwrite_t<Tu>& R, 
  const Ti r_offs,
  const Ti r_size
);

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
  
} // namespace boundary
  
#include <boundary.ipp>
#endif
