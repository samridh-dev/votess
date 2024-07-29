#ifndef VOTESS_HPP
#define VOTESS_HPP

#include <vector>
#include <array>

namespace votess {

struct vtargs;

enum device { cpu, gpu, };

template <typename Ti> class dnn;

template <typename Ti, typename Tf>
class dnn<Ti> 
tesellate(
  std::vector<std::array<Tf,3>>& xyzset,
  const struct vtargs& args,
  const enum device device = device::gpu
);

} // namespace votess
  
#include <votess.ipp>
#endif // votess.hpp
