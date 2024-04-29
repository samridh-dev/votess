#include <utils.hpp>
template <typename T2>
inline T2 sr::update(
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2,
  const T2 security_radius
) {
  const T2 dx = utils::square(x1 - x2);
  const T2 dy = utils::square(y1 - y2);
  const T2 dz = utils::square(z1 - z2);
  const T2 new_radius = dx + dy + dz;
  const bool cond = new_radius > security_radius;
  const T2 val = (!cond) * security_radius + (cond) * new_radius;
  return val;
}

#include <utils.hpp>
template <typename T2>
inline bool sr::is_reached(
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2,
  const T2 security_radius
) {
  const T2 dx = utils::square(x1 - x2);
  const T2 dy = utils::square(y1 - y2);
  const T2 dz = utils::square(z1 - z2);
  const T2 radius  = dx + dy + dz;
  const bool cond = radius > 4.0f * security_radius;
  return cond;
}
