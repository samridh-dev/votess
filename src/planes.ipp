template <typename T2>
inline void planes::intersect(
  T2& dst0, T2& dst1, T2& dst2, T2& dst3,
  const T2 a1, const T2 b1, const T2 c1, const T2 d1,
  const T2 a2, const T2 b2, const T2 c2, const T2 d2,
  const T2 a3, const T2 b3, const T2 c3, const T2 d3
) {

  const T2 n12_a =   b1 * c2 - b2 * c1;
  const T2 n12_b = - a1 * c2 + a2 * c1;
  const T2 n12_c =   a1 * b2 - a2 * b1;

  const T2 n23_a =   b2 * c3 - b3 * c2;
  const T2 n23_b = - a2 * c3 + a3 * c2;
  const T2 n23_c =   a2 * b3 - a3 * b2;

  const T2 n31_a =   b3 * c1 - b1 * c3;
  const T2 n31_b = - a3 * c1 + a1 * c3;
  const T2 n31_c =   a3 * b1 - a1 * b3;

  T2 triple_product = a1 * n23_a + b1 * n23_b + c1 * n23_c; 
  triple_product = (triple_product == 0) ? 0 : 1.0f / triple_product;

  const T2 dst0_coefficient = - d1 * n23_a - d2 * n31_a - d3 * n12_a;
  const T2 dst1_coefficient = - d1 * n23_b - d2 * n31_b - d3 * n12_b;
  const T2 dst2_coefficient = - d1 * n23_c - d2 * n31_c - d3 * n12_c;
  const T2 dst3_coefficient = 1;

  dst0 = dst0_coefficient * triple_product;
  dst1 = dst1_coefficient * triple_product;
  dst2 = dst2_coefficient * triple_product;
  dst3 = dst3_coefficient;

}
 
template <typename T2>
inline T2 planes::dot(
  const T2 a1, const T2 b1, const T2 c1, const T2 d1,
  const T2 a2, const T2 b2, const T2 c2, const T2 d2
) {
  const T2 dot0 = a1 * a2;
  const T2 dot1 = b1 * b2;
  const T2 dot2 = c1 * c2;
  const T2 dot3 = d1 * d2;
  const T2 dotp = dot0 + dot1 + dot2 + dot3;
  return dotp;
}

#include <utils.hpp>
template <typename T2>
inline void planes::bisect(
  T2& dst0, T2& dst1, T2& dst2, T2& dst3,
  const T2 x1, const T2 y1, const T2 z1,
  const T2 x2, const T2 y2, const T2 z2
) {
  const T2 xsquare = (utils::square(x1) - utils::square(x2));
  const T2 ysquare = (utils::square(y1) - utils::square(y2));
  const T2 zsquare = (utils::square(z1) - utils::square(z2));
  dst0 = x1 - x2;   
  dst1 = y1 - y2;   
  dst2 = z1 - z2;   
  dst3 = - (xsquare + ysquare + zsquare) / 2;
}
