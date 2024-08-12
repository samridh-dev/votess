#ifndef VTARGS_HPP
#define VTARGS_HPP

// Using macros for two reasons:
//  * pybind11 compatibility
//  * ability to have mixed types

#ifndef ARGS_DEFAULT_K
#define ARGS_DEFAULT_K 64
#endif

#ifndef ARGS_DEFAULT_CPU_NTHREADS
#define ARGS_DEFAULT_CPU_NTHREADS 0
#endif

#ifndef ARGS_DEFAULT_GPU_NDWORKSIZE
#define ARGS_DEFAULT_GPU_NDWORKSIZE 1
#endif

#ifndef ARGS_DEFAULT_CHUNKSIZE
#define ARGS_DEFAULT_CHUNKSIZE 8196
#endif

#ifndef ARGS_DEFAULT_USE_CHUNKING
#define ARGS_DEFAULT_USE_CHUNKING false
#endif

#ifndef ARGS_DEFAULT_USE_RECOMPUTE
#define ARGS_DEFAULT_USE_RECOMPUTE false
#endif

#ifndef ARGS_DEFAULT_GRID_RESOLUTION
#define ARGS_DEFAULT_GRID_RESOLUTION 16
#endif

#ifndef ARGS_DEFAULT_P_MAXSIZE
#define ARGS_DEFAULT_P_MAXSIZE 32
#endif

#ifndef ARGS_DEFAULT_T_MAXSIZE
#define ARGS_DEFAULT_T_MAXSIZE 32
#endif

#include <string>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

#include <initializer_list>

namespace args {

struct xyzset {
  int grid_resolution;
  xyzset(const int gr0) : grid_resolution(gr0) {}
};

struct knn {
  int k;
  int grid_resolution;
  knn(const int k0, const int gr0) : k(k0), grid_resolution(gr0) {}
};

struct cc {
  int k;
  int p_maxsize;
  int t_maxsize;
  cc(const int k0, const int pms0, const int tms0) 
    : k(k0), p_maxsize(pms0), t_maxsize(tms0) {}
};

} // namespace args

namespace votess {

class vtargref {
  private:
    std::string str;

    template <typename T>
    std::string to_str(const T v) const {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    }

    template <typename T>
    T from_str(const std::string& s) const {
      std::istringstream iss(s); 
      T v;
      iss >> v;
      if (iss.fail()) {
        throw std::invalid_argument("Invalid type conversion");
      }
      return v;
    }
  
  public:

    template <typename T> 
    vtargref& operator=(const T& other) {
      this->str = to_str(other);
      return *this;
    }

    template <typename T>
    operator T() const {
      return from_str<T>(str);
    }

    template<typename T> 
    T get() const {
      return from_str<T>(str);
    }

};

class vtargs {
  private:
    std::unordered_map<std::string, vtargref> map;

    void init(void) {
      map["k"] = ARGS_DEFAULT_K;

      map["cpu_nthreads"] = ARGS_DEFAULT_CPU_NTHREADS;
      map["gpu_ndsize"] = ARGS_DEFAULT_GPU_NDWORKSIZE;

      map["chunksize"] = ARGS_DEFAULT_CHUNKSIZE;

      map["use_chunking"] = ARGS_DEFAULT_USE_CHUNKING;
      map["use_recompute"] = ARGS_DEFAULT_USE_RECOMPUTE;

      map["knn_grid_resolution"] = ARGS_DEFAULT_GRID_RESOLUTION;

      map["cc_p_maxsize"] = ARGS_DEFAULT_P_MAXSIZE;
      map["cc_t_maxsize"] = ARGS_DEFAULT_T_MAXSIZE;

      map["dev_suppress_stdout"] = true;

    }

  public:

    vtargs(void) {
      init();
    }

  vtargref& operator[](const std::string& key) {
    auto it = map.find(key);
    if (it == map.end()) {
      throw std::invalid_argument("key: " + key + 
                                  " not found in file: " + __FILE__ + 
                                  " at line: " + std::to_string(__LINE__));
    }
    return it->second;
  }

  const vtargref& operator[](const std::string& key) const {
    auto it = map.find(key);
    if (it == map.end()) {
      throw std::invalid_argument("key: " + key + 
                                  " not found in file: " + __FILE__ + 
                                  " at line: " + std::to_string(__LINE__));
    }
    return it->second;
  }

    const struct args::xyzset get_xyzset(void) const {
      int gr = (*this)["knn_grid_resolution"];
      return args::xyzset(gr);
    }

    const struct args::knn get_knn(void) const {
      int gr = (*this)["knn_grid_resolution"];
      int k = (*this)["k"];
      return args::knn(k, gr);
    }

    const struct args::cc get_cc(void) const {
      int k = (*this)["k"];
      int p_maxsize = (*this)["cc_p_maxsize"];
      int t_maxsize = (*this)["cc_t_maxsize"];
      return args::cc(k, p_maxsize, t_maxsize);
    }

};

} // namespace votess

#endif // VTARGS_HPP
