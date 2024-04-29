#ifndef PLANES_HPP
#define PLANES_HPP
#include <array>

namespace planes {

///////////////////////////////////////////////////////////////////////////////
/// Plane Functions
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T2>
inline void intersect(
  T2& a0, T2& b0, T2& c0, T2& d0,
  const T2 a1, const T2 b1, const T2 c1, const T2 d1,
  const T2 a2, const T2 b2, const T2 c2, const T2 d2,
  const T2 a3, const T2 b3, const T2 c3, const T2 d3
); 

/* ------------------------------------------------------------------------- */

template <typename T2>
inline T2 dot(
  const T2 a1, const T2 b1, const T2 c1, const T2 d1,
  const T2 a2, const T2 b2, const T2 c2, const T2 d2
);

/* ------------------------------------------------------------------------- */

template <typename T2>
inline void bisect(
  T2& dst0, T2& dst1, T2& dst2, T2& dst3,
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2
);

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////

} // namespace planes

#include <planes.ipp>

#endif
