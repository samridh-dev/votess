#ifndef ARGUMENTS_HPP
#define ARGUMENTS_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

#define USE_NEW_ARGS 0

///////////////////////////////////////////////////////////////////////////////
/// Compile time default parameters
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

// using macros for two reasons:
//  * pybind11 compatibility
//  * ability to have mixed types

#define ARGS_DEFAULT_P_MAXSIZE 128
#define ARGS_DEFAULT_T_MAXSIZE 128

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Function specific arguments (internal)
///////////////////////////////////////////////////////////////////////////////
namespace args {

/* ------------------------------------------------------------------------- */

struct global {
  int k;
  int __cpu__nthreads;
  int __gpu__ndworksize;
  global(const int k0) : 
    k(k0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct xyzset {
  int grid_resolution;
  xyzset(const int gr0) : 
    grid_resolution(gr0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct knn {
  int k;
  int grid_resolution;
  knn(const int k0, const int gr0) : 
    k(k0),
    grid_resolution(gr0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct cc {
  int k;
  int p_maxsize;
  int t_maxsize;
  cc(const unsigned short int k0,
      const unsigned short int pms0, const unsigned short int tms0
  ) : k(k0), p_maxsize(pms0), t_maxsize(tms0) {}
};

/* ------------------------------------------------------------------------- */
} // namespace args

namespace votess {

#if USE_NEW_ARGS
class vtargs {

  private:
    
    // global
    int k;
    int __cpu__nthreads;
    int __gpu__ndsize;
    int chunksize;
    bool use_recompute;
    
    // knn
    int gr;

    // dnn
    int p_maxsize;
    int t_maxsize;

    struct args::xyzset xyzset;
    struct args::knn knn;
    struct args::cc cc;
  
    using arg_t std::variant<int, bool>
    std::unordered_map<std::string, arg_t> map;

    void init(void);
    void update(void);

  public:

    vtargs(void);
    vtargs(const int _k);

    arg_t operator[](const std::string& key) const;
    arg_t& operator[](const std::string& key);
    void operator()(const std::string& key, const arg_t& val);
  
};
} // namespace votess

void votess::vtargs::init(void) {
  map["k"] = k;
  map["cpu_nthread"] = __cpu__nthreads;
  map["gpu_nd_worksize"] = __gpu__ndsize;
  map["chunk_size"] = chunksize;
  map["allow_recompute"] = chunksize;
  map["knn_grid_resolution"] = gr;
  map["dnn_p_maxsize"] = p_maxsize;
  map["dnn_t_maxsize"] = t_maxsize;
}

votess::vtargs::vtargs(void) {
  init();
}

votess::vtargs::vtargs(const int _k) {
  init();
  k = _k;
}

arg_t votess::vtargs::operator[](const std::string& key) const {
  auto it = map.find(key);
  if (it != map.end()) {
    return it->second;
  } else {
    throw std::invalid_argument("Key not found");
  }
}

arg_t& votess::vtargs::operator[](const std::string& key) {
  auto it = map.find(key);
  if (it == map.end()) {
    throw std::invalid_argument("Key not found");
  }
  return it->second;
}

void votess::vtargs::operator()(const std::string& key, const arg_t& val) {
  auto it = map.find(key);
  if (it != map.end()) {
    it->second = val;
    update();
  } else {
    throw std::invalid_argument("Key not found");
  }
}

#else
struct vtargs {

  struct args::global global;
  struct args::xyzset xyzset;
  struct args::knn knn;
  struct args::cc cc;

  const size_t nthreads;

  vtargs(
    const int _k,
    const int _grid_resolution = 1,
    const int _nthreads = 1,
    const int _p_maxsize = ARGS_DEFAULT_P_MAXSIZE,
    const int _t_maxsize = ARGS_DEFAULT_T_MAXSIZE) 
    : global(_k),
      xyzset(_grid_resolution),
      knn(_k, _grid_resolution),
      cc(_k, _p_maxsize , _t_maxsize),
      nthreads(_nthreads) {}

  void set_k(const int k) {
    global.k = k;
    knn.k = k;
    cc.k = k;
  }

};
} // namespace votess
#endif // USE_NEW_ARGS


#endif
