#ifndef SECURITY_RADIUS_HPP
#define SECURITY_RADIUS_HPP

namespace sr {
  
///////////////////////////////////////////////////////////////////////////////
/// Security Radius functions
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T2>
inline T2 update(
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2,
  const T2 radius
);

/* ------------------------------------------------------------------------- */
  
template <typename T2>
inline bool is_reached(
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2,
  const T2 security_radius
);

/* ------------------------------------------------------------------------- */
  
///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
  
} // namespace sr
#include <sradius.ipp>
#endif
