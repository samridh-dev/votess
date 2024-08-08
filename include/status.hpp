#ifndef STATUS_HPP
#define STATUS_HPP

/* ------------------------------------------------------------------------- */
/// Macros
/* ------------------------------------------------------------------------- */

#define __INTERNAL__K_UNDEFINED -32

/* ------------------------------------------------------------------------- */

namespace cc {

///////////////////////////////////////////////////////////////////////////////
/// Voronoi Cell Status Struct
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

enum status : signed char {
  security_radius_reached,
  error_infinite_boundary,
  error_nonvalid_vertices,
  error_nonvalid_neighbor,
  error_p_overflow,
  error_t_overflow,
  error_occurred,

  status_enum_size,
  k_undefined = __INTERNAL__K_UNDEFINED
};

/* ------------------------------------------------------------------------- */

struct state {
  uint8_t byte;

  inline state();
  inline bool get(const enum status s) const;
  inline void set_true(const enum status s);
  inline void set_false(const enum status s);
  inline void reset();

};

/* ------------------------------------------------------------------------- */
  
///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
  
} // namespace cc
  
#include <status.ipp>
#endif
