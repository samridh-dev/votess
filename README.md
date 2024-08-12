# Votess

Votess is a library to perform 3D Voronoi tessellation computations. It
leverages the SYCL framework, giving it the ability to run on heterogenous
platforms, including GPUs and FPGAs. It was developed to be parallel,
performant, but portable, and was also developed for applications only
requiring the geometry of the voronoi diagram, for example in cosmological
simulations, or K-means clustering.

This library is part of my bachelor project, which was only made possible
through the guidance of Dr. Chris Byrohl and Dr. Dylan Nelson from the Institute
of Theoretical Astrophysics at Heidelberg University, and for that I owe my
gratitude to them.

## Dependencies

The project requires a SYCL-compatible compiler supporting C++17 or later.
Currently, the following compilers have been tested and confirmed to work:

- **Intel® oneAPI DPC++/C++ Compiler** (Recommended)
- **AdaptiveCpp (formerly known as hipSYCL / Open SYCL) acpp Compiler** 

The library and command line program does **not** require any additional
dependencies. However, if you wish to compile the python bindings or the test
cases, the following dependencies are required.

#### Python Bindings
- [**pybind11**:](https://github.com/pybind/pybind11)

#### Test Cases
- [**voro++**:](https://github.com/chr1shr/voro)
- [**Catch2**:](https://github.com/catchorg/Catch2)

In any case, the build script will manage all dependencies locally so manual
installation is not necessary.

## Installation

Currently compilation from source is the only possibility.

### Compiler Installation.

To install the Intel® oneAPI compiler, refer to the installation
[instructions](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html).

If using Arch Linux, the following command will be sufficient:
```bash
pacman -S intel-oneapi-basekit
```

Before using the compiler, it is necessary to source the relevant environment
variables. Instructions on how to do so are provided for:
 - [Linux](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2024-1/get-started-on-linux.html) 
 - [Windows](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2024-1/get-started-on-windows.html).

To install AdaptiveCpp, refer to the installation page on Github 
[here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md).

### Application Compilation
To get the repository, first clone the repository and switch to the cloned 
directory

```bash
git clone https://github.com/samridh-dev/votess.git
cd votess
```
you should then review the compilation options located in the file
`${project-root}/cmake/options.cmake`

After that, execute the command located at project root:

```bash
./make.sh
```

All binaries will then be generated at `${project-root}/bin/`

# Usage

There exists three APIs one could use: 
`Python >=3.5, >=C++17, (Unix)*sh`

If compiled with Intel® oneAPI DPC++/C++ Compiler, sourcing environment
variables before running the application is required. 

### C++

The C++ API comes with the following templated function:
```cpp
namespace votess {
  template <typename Ti, typename Tf>
  class dnn<Ti> tesellate(
    std::vector<std::array<Tf,3>>& xyzset,
    class vtargs args,
    const enum device device = device::cpu
  );
}
```

The parameter `xyzset` represents the 3 dimensional array of floating points
values set in a row major format. It is a strict requirement that the
underlying type `xyzset` is floating point, and that each value lies between 0
and 1 (exclusive).  The parameter `device` is used to set the device to run the
tessellation on, and args is used to set parameters in the function.  The usage
of `class vtargs` is similar to that of `std::unordered_map`. 

For example:

```cpp
class vtargs vtargs;
vtargs["k"] = 64;              // arithmetic types are accepted
vtargs["use_chunking"] = true; // booleans are accepted for certain parameters
```

A list of all possible parameters is provided in the table below:

| Parameter Name         | Description                                                                               |
|------------------------|-------------------------------------------------------------------------------------------|
| `k`                    | Number of nearest neighbors                                                               |
| `cpu_nthreads`         | Number of CPU threads to use. Set to 0 for highest thread count available in the machine  |
| `gpu_ndsize`           | GPU work size. Recommended to set in multiples of 16                                      |
| `use_recompute`        | Set to `true` to enable CPU fallback. This will ensure all points are valid voronoi cells |
| `use_chunking`         | Set to `true` to split processing in chunks.                                              |
| `chunksize`            | Size of chunks for processing. Set a small value for the CPU, and a large one for the GPU |
| `knn_grid_resolution`  | Grid resolution for k-nearest-neighbors algorithm                                         |
| `cc_p_maxsize`         | Maximum size of P parameter for convex cell algorithm                                     |
| `cc_t_maxsize`         | Maximum size of T parameter for convex cell algorithm                                     |
| `dev_suppress_stdout`  | Developer parameter to enable stdout. Defaults to `false`                                 |

The return type `class dnn` is a jagged 2 dimensional array representing the
nearest neighbors that directly contribute to the voronoi cell. It has been
written to have a contiguous memory layout, so that accesses would be more
efficient. Do note that the ordering of xyzset is **not** preserved, as it will
be sorted during the tessellation. **When the interface is properly implemented
will it then be documented. for now use the examples below.**

An example program to tessellate a point set is given below:

### Examples 

```cpp
#include <votess.hpp>

// initial number of neighbors per point
const int k = 9;

// grid fineness for knn algorithm
const int grid_resolution = 2;

int main(int argc, char* argv[]) {

  // sample dataset
  std::vector<std::array<float, 3>> xyzset = {
    {0.605223f, 0.108484f, 0.0909372f},
    {0.500792f, 0.499641f, 0.464576f},
    {0.437936f, 0.786332f, 0.160392f},
    {0.663354f, 0.170894f, 0.810284f},
    {0.614869f, 0.0968678f, 0.204147f},
    {0.556911f, 0.895342f, 0.802266f},
    {0.305748f, 0.124146f, 0.516249f},
    {0.406888f, 0.157835f, 0.919622f},
    {0.0944123f, 0.861991f, 0.798644f},
    {0.511958f, 0.560537f, 0.345479f}
  };

  // votess::tesellate parameter struct
  struct votess::vtargs vtargs();
  vtargs["k"] = k;
  vtargs["knn_grid_resolution"] = grid_resolution;

  // get direct neighbors that constitute a voronoi cell for each point.
  auto dnn = votess::tesellate<int, float>(xyzset, vtargs);

  // Alternatively, get direct neighbors via cpu 
  // auto dnn = votess::tesellate<int, float>(xyzset, vtargs, 
  //                                          votess::device::cpu);
    
  // use direct_neighbors as a normal 2 dimensional array
  std::cout << "First neighbor for point 0: " << dnn[0][0] << std::endl;
  std::cout << "number of direct neighbors for point 0: "
            << dnn[0].size()
            << std::endl;

  return 0;
}
```

## Python

Similarly to the C++ implementation, a python wrapper library named `pyvotess`
exists. The API is similar to that of votess, but can also leverage the numpy
library.

### Example Usage
```python
import pyvotess as vt
import numpy as np

k = 9
grid_resolution = 1

xyzset = np.array([
  [0.605223, 0.108484, 0.0909372],
  [0.500792, 0.499641, 0.464576],
  [0.437936, 0.786332, 0.160392],
  [0.663354, 0.170894, 0.810284],
  [0.614869, 0.0968678, 0.204147],
  [0.556911, 0.895342, 0.802266],
  [0.305748, 0.124146, 0.516249],
  [0.406888, 0.157835, 0.919622],
  [0.0944123, 0.861991, 0.798644],
  [0.511958, 0.560537, 0.345479]
])

vtargs = vt.vtargs()
vtargs["k"] = k
vtargs["knn_grid_resolution"] = grid_resolution 

direct_neighbors = vt.tesellate(xyzset, vtargs)

# Alternatively, get direct neighbors via cpu
# direct_neighbors = vt.tesellate(xyzset, vtargs, vt.device.cpu)

# use direct_neighbors as a normal 2 dimensional array
print("First neigbhor for point 0: ", direct_neighbors[0][0])
print("size of direct neighbors for point 0", direct_neighbors[0].size())

```
### Command Line

There exists the executable named `clvotess`. 
**Currently the program is undeveloped as the project is still undergoing
changes. Usage will remain undocumented at the time being.**
