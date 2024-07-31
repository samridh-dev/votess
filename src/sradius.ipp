#include <utils.hpp>
template <typename Tf>
inline Tf sr::update(
  const Tf x1, const Tf y1, const Tf z1,
  const Tf x2, const Tf y2, const Tf z2,
  const Tf security_radius
) {
  const Tf dx = utils::square(x1 - x2);
  const Tf dy = utils::square(y1 - y2);
  const Tf dz = utils::square(z1 - z2);
  const Tf new_radius = dx + dy + dz;
  const bool cond = new_radius > security_radius;
  const Tf val = (!cond) * security_radius + (cond) * new_radius;
  return val;
}

#include <utils.hpp>
template <typename Tf>
inline bool sr::is_reached(
  const Tf x1, const Tf y1, const Tf z1,
  const Tf x2, const Tf y2, const Tf z2,
  const Tf security_radius
) {
  const Tf dx = utils::square(x1 - x2);
  const Tf dy = utils::square(y1 - y2);
  const Tf dz = utils::square(z1 - z2);
  const Tf radius  = dx + dy + dz;
  const bool cond = radius > 4.0f * security_radius;
  return cond;
}
