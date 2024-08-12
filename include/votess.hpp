#ifndef VOTESS_HPP
#define VOTESS_HPP

#include <vector>
#include <array>

#include <arguments.hpp>
#include <dnn.hpp>

namespace votess {
  enum device { cpu, gpu, };
}

namespace votess {
  template <typename Ti, typename Tf>
  class dnn<Ti> tesellate(
    std::vector<std::array<Tf, 3>>& xyzset,
    class vtargs args,
    const enum device device = device::cpu
  );
}
  
#include <votess.ipp>
#endif // votess.hpp
