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

template<typename T1>
inline bstatus compute(
  T1* cycle, 
  const size_t dr_offs,
  const size_t dr_size,
  short int& head,
  T1* R,
  const size_t r_offs,
  const size_t r_size
);

/* ------------------------------------------------------------------------- */
/// SYCL implementation
/* ------------------------------------------------------------------------- */

template<typename T3>
inline bstatus compute(
  const device_accessor_readwrite_t<T3>& cycle, 
  const size_t dr_offs,
  const size_t dr_size,
  short int& head,
  const device_accessor_readwrite_t<T3>& R, 
  const size_t r_offs,
  const size_t r_size
);

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
  
} // namespace boundary
  
#include <boundary.ipp>
#endif
