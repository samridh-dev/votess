#ifndef VOTESS_HPP
#define VOTESS_HPP

#include <arguments.hpp>
#include <vector>
#include <array>
#include <cstdint>

namespace votess {
  
///////////////////////////////////////////////////////////////////////////////
/// Direct neighbor class
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class dnn {
  public:
    class proxy;

    dnn();
    dnn(std::vector<T>& _list, std::vector<T>& _offs);

    const proxy operator[](const int i) const;
    proxy operator[](const int i);
    size_t size() const;
    
    void print() const;
    void savetxt(const std::string& fname) const;

  private:
    std::vector<T> list;
    std::vector<T> offs;
};

///////////////////////////////////////////////////////////////////////////////
/// Tesellate function
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// Device List
/* ------------------------------------------------------------------------- */

enum device { cpu, gpu, };

/* ------------------------------------------------------------------------- */
/// Tesellate Parameters
/* ------------------------------------------------------------------------- */

struct vtargs {

  struct args::global global;
  struct args::xyzset xyzset;
  struct args::knn knn;
  struct args::cc cc;

  const size_t nthreads;

  vtargs(
    const int _k,
    const int _grid_resolution,
    const int _nthreads = 1,
    const int _p_maxsize = ARGS_DEFAULT_P_MAXSIZE, 
    const int _t_maxsize = ARGS_DEFAULT_T_MAXSIZE
  ) : global(_k), 
      xyzset(_grid_resolution),
      knn(_k, _grid_resolution), 
      cc(_k, _p_maxsize , _t_maxsize),
      nthreads(_nthreads)
  {}

};

/* ------------------------------------------------------------------------- */
/// Main Function
/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
class dnn<T1>
tesellate(
  std::vector<std::array<T2,3>>& xyzset,
  const struct vtargs& args,
  const enum device device = device::gpu
);

/* ------------------------------------------------------------------------- */
/// Device Specific Implementations
/* ------------------------------------------------------------------------- */

namespace dtessellate {

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
class dnn<T1>
cpu(std::vector<std::array<T2,3>>& xyzset, const struct vtargs& args);

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
class dnn<T1>
gpu(std::vector<std::array<T2,3>>& xyzset, const struct vtargs& args);

/* ------------------------------------------------------------------------- */
} // namespace tesellate

///////////////////////////////////////////////////////////////////////////////
/// end
///////////////////////////////////////////////////////////////////////////////

} // namespace votess
#include <votess.ipp>
#endif
