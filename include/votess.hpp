#ifndef VOTESS_HPP
#define VOTESS_HPP

#include <vector>
#include <array>

namespace votess {

enum device { cpu, gpu, };

template <typename Ti> class dnn;

template <typename Ti, typename Tf>
class dnn<Ti> 
tesellate(
  std::vector<std::array<Tf,3>>& xyzset,
  struct vtargs args,
  const enum device device = device::cpu
);

} // namespace votess
  
#include <votess.ipp>
#endif // votess.hpp
