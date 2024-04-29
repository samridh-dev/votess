#ifndef ARGUMENTS_HPP
#define ARGUMENTS_HPP

#include <iostream>

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
  unsigned short int k;
  global(const int k0) : 
    k(k0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct xyzset {
  unsigned short int grid_resolution;
  xyzset(const int gr0) : 
    grid_resolution(gr0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct knn {
  unsigned short int k;
  unsigned short int grid_resolution;
  knn(const int k0, const int gr0) : 
    k(k0),
    grid_resolution(gr0) 
  {}
};

/* ------------------------------------------------------------------------- */

struct cc {
  unsigned short int k;
  unsigned short int p_maxsize;
  unsigned short int t_maxsize;
  cc(const unsigned short int k0,
      const unsigned short int pms0, const unsigned short int tms0
  ) : k(k0), p_maxsize(pms0), t_maxsize(tms0) {}
};

/* ------------------------------------------------------------------------- */
} // namespace args

#endif
