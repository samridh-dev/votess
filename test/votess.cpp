#include <catch2/catch_test_macros.hpp> 
#include <catch2/matchers/catch_matchers_floating_point.hpp> 

#include <iostream>

#include <votess.hpp>

///////////////////////////////////////////////////////////////////////////////
/// Forward Declarations
///////////////////////////////////////////////////////////////////////////////

template <typename T>
static void test_votess(
  std::vector<std::array<T,3>>& xyzset,
  const int k,
  const int gr
);

///////////////////////////////////////////////////////////////////////////////
/// Test Cases                                                                
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 1: standard", "[votess]") {

  const int k = 9;
  std::vector<std::array<float, 3>> xyzset = {
    {0.605223f, 0.108484f, 0.090937f}, {0.500792f, 0.499641f, 0.464576f},
    {0.437936f, 0.786332f, 0.160392f}, {0.663354f, 0.170894f, 0.810284f},
    {0.614869f, 0.096867f, 0.204147f}, {0.556911f, 0.895342f, 0.802266f},
    {0.305748f, 0.124146f, 0.516249f}, {0.406888f, 0.157835f, 0.919622f},
    {0.094412f, 0.861991f, 0.798644f}, {0.511958f, 0.560537f, 0.345479f}
  };

  for (const auto& gr : {1, 2, 3, 4, 6, 8, 16, 24, 32} ) {
    test_votess(xyzset, k, gr);
  }
}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 2: clustured", "[votess]") {

  const int k = 8;
  std::vector<std::array<float, 3>> xyzset = {
    {0.1f, 0.2f, 0.3f}, {0.11f, 0.19f, 0.31f}, {0.09f, 0.21f, 0.29f},
    {0.5f, 0.5f, 0.5f}, {0.51f, 0.49f, 0.51f}, {0.49f, 0.51f, 0.49f},
    {0.8f, 0.8f, 0.8f}, {0.81f, 0.79f, 0.81f}, {0.79f, 0.81f, 0.79f} 
  };

  for (const auto& gr : {1, 2, 3, 4, 6, 8, 16, 24, 32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 3: uniform", "[votess]") {

  const int k = 8;
  std::vector<std::array<float, 3>> xyzset = {
    {0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f}, {0.3f, 0.3f, 0.3f},
    {0.4f, 0.4f, 0.4f}, {0.5f, 0.5f, 0.5f}, {0.6f, 0.6f, 0.6f},
    {0.7f, 0.7f, 0.7f}, {0.8f, 0.8f, 0.8f}, {0.9f, 0.9f, 0.9f}
  };

  for (const auto& gr : {1, 2, 3, 4, 6, 8, 16, 24, 32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 4: colinear points", "[votess]") {

  const int k = 2;
  std::vector<std::array<float, 3>> xyzset = {
    {0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f}, {0.3f, 0.3f, 0.3f}
  };

  for (const auto& gr : {1, 2, 3, 4, 5, 8, 16, 24, 32} ) {
    test_votess(xyzset, k, gr);
  }

}
/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 5: line distribution 1", "[votess]") {

  const int k = 8;
  std::vector<std::array<float, 3>> xyzset = {
    {0.1f, 0.5f, 0.5}, {0.2f, 0.5f, 0.5}, {0.3f, 0.5f, 0.5},
    {0.4f, 0.5f, 0.5}, {0.5f, 0.5f, 0.5}, {0.6f, 0.5f, 0.5},
    {0.7f, 0.5f, 0.5}, {0.8f, 0.5f, 0.5}, {0.9f, 0.5f, 0.5},
  };

  for (const auto& gr : {1, 2, 3, 4, 5, 8, 16, 24, 32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 6: fibonnacci sphere", "[votess]") {

  const int k = 15;
  std::vector<std::array<float, 3>> xyzset = {
    {0.500000f, 0.750000f, 0.500000f}, {0.408034f, 0.716667f, 0.584248f},
    {0.514860f, 0.683333f, 0.330683f}, {0.621688f, 0.650000f, 0.658720f},
    {0.282272f, 0.616667f, 0.461487f}, {0.698875f, 0.583333f, 0.373492f},
    {0.436410f, 0.550000f, 0.736551f}, {0.385030f, 0.516667f, 0.278631f},
    {0.734308f, 0.483333f, 0.585569f}, {0.273583f, 0.450000f, 0.593462f},
    {0.599901f, 0.416667f, 0.286516f}, {0.566174f, 0.383333f, 0.710974f},
    {0.326958f, 0.350000f, 0.399718f}, {0.666003f, 0.316667f, 0.463505f},
    {0.428269f, 0.283333f, 0.602030f}, {0.500000f, 0.250000f, 0.500000f}
  };

  for (const auto& gr : {1,2,3,4,5,8,16,24,32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 7: coplanar points", "[votess]") {

  const int k = 3;
  std::vector<std::array<float, 3>> xyzset = {
    {0.2f, 0.2f, 0.1f}, {0.3f, 0.3f, 0.1f}, 
    {0.4f, 0.4f, 0.1f}, {0.5f, 0.5f, 0.1f}
  };

  for (const auto& gr : {1,2,3,4,5,8,16,24,32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

TEST_CASE("votess regression 8: concentric points", "[votess]") {

  const int k = 4;
  std::vector<std::array<float, 3>> xyzset = {
    {0.5, 0.5, 0.5},  {0.55, 0.5, 0.5},
    {0.5, 0.55, 0.5}, {0.45, 0.5, 0.5},
    {0.5, 0.45, 0.5}
  };

  for (const auto& gr : {1,2,3,4,5,8,16,24,32} ) {
    test_votess(xyzset, k, gr);
  }

}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Helper functions
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

class __internal__suppress_stdout {
  public:
    __internal__suppress_stdout() : buf(std::cout.rdbuf()) {
      std::cout.rdbuf(__tmp__buf.rdbuf());
    }
    ~__internal__suppress_stdout() {
      std::cout.rdbuf(buf);
    }
  private:
    std::streambuf* buf;
    std::stringstream __tmp__buf;
};

/* ------------------------------------------------------------------------- */

template <typename T>
static void 
save(const std::vector<std::array<T,3>>& vec, const std::string& fname) {
  std::ofstream fout(fname);
  for (const auto& arr : vec) fout<<arr[0]<<" "<<arr[1]<<" "<<arr[2]<<"\n";
}

/* ------------------------------------------------------------------------- */

#include <voro++.hh>
#include <filesystem>
#include <vector>
#include <array>
#include <utility>

template <typename T>
static std::pair<std::vector<std::array<T, 3>>, std::vector<std::vector<int>>>
run_voro(
  const std::vector<std::array<T,3>>& xyzset,
  const struct votess::vtargs vtargs
) {
  std::vector<std::array<T, 3>> coords;
  std::vector<std::vector<int>> neighbor_list;
  const double tolerance = 1e-8;

  using namespace voro;
  container con(
    0, 1, 0, 1, 0, 1,
    vtargs.xyzset.grid_resolution,
    vtargs.xyzset.grid_resolution,
    vtargs.xyzset.grid_resolution,
    false, false, false,
    xyzset.size()
  );
  
  for (size_t i = 0; i < xyzset.size(); i++) {
    con.put(i,xyzset[i][0],xyzset[i][1], xyzset[i][2]);
  }

  c_loop_all cl(con);
  voronoicell_neighbor c;
  if (cl.start()) do if (con.compute_cell(c, cl)) {
    std::vector<int> neighbors;
    std::vector<int> filtered_neighbors;
    std::vector<double> face_areas;
    double x, y, z;

    cl.pos(x, y, z);
    c.neighbors(neighbors);
    c.face_areas(face_areas);

    for (size_t i = 0; i < face_areas.size(); i++) {
      if (face_areas[i] >= tolerance) {
        filtered_neighbors.push_back(neighbors[i]);
      }
    }

    neighbor_list.push_back(filtered_neighbors);
    coords.push_back({
      static_cast<T>(x),
      static_cast<T>(y),
      static_cast<T>(z)
    });

  } while(cl.inc());

  return {coords, neighbor_list};
}

/* ------------------------------------------------------------------------- */


template <typename T>
static void run_test(
  std::vector<std::array<T,3>>& xyzset,
  struct votess::vtargs vtargs,
  const enum votess::device device
) {
    __internal__suppress_stdout s; // to preventstdout 

    (void)xyzset::sort<int,T>(xyzset, vtargs.xyzset);

    auto [vcoord, vneighbor] = run_voro<T>(xyzset, vtargs);
    auto dnn = votess::tesellate<int, T>(xyzset, vtargs);

    std::vector<int> test_dnn(0);
    std::vector<int> test_vneighbor(0);

    for (size_t i = 0; i < xyzset.size(); i++) {
        auto it = std::find_if(
            vcoord.begin(),
            vcoord.end(),
            [&](const std::array<T, 3>& elem) {
                return (elem[0] == xyzset[i][0]) &&
                       (elem[1] == xyzset[i][1]) &&
                       (elem[2] == xyzset[i][2]);
            });

        if (it == vcoord.end()) {
            CAPTURE(i);
            WARN("Matching coordinate not found in vcoord");
            continue;
        }

        const size_t index = std::distance(vcoord.begin(), it);
        CAPTURE(index);

        test_dnn.clear();
        test_vneighbor.clear();
        for (const auto& j : vneighbor[index]) {
            if (j < 0) continue;
            test_vneighbor.push_back(j);
        }
        for (size_t j = 0; j < dnn[i].size(); j++) {
            test_dnn.push_back(dnn[i][j]);
        }

        std::sort(test_vneighbor.begin(), test_vneighbor.end());
        std::sort(test_dnn.begin(), test_dnn.end());

        CAPTURE(test_vneighbor, test_dnn);

        bool cond = std::includes(
          test_dnn.begin(),
          test_dnn.end(),
          test_vneighbor.begin(),
          test_vneighbor.end()
        );
        CAPTURE(cond);
        REQUIRE(cond);
    }
}

/* ------------------------------------------------------------------------- */

template <typename T>
static void test_votess(
  std::vector<std::array<T,3>>& xyzset,
  const int k,
  const int gr
) {
  SECTION("(CPU) case : grid_resolution = " + std::to_string(gr)) {
    struct votess::vtargs vtargs(k,gr);
    run_test(xyzset, vtargs, votess::device::cpu);
  }
  SECTION("(GPU) case : grid_resolution = " + std::to_string(gr)) {
    struct votess::vtargs vtargs(k,gr);
    run_test(xyzset, vtargs, votess::device::gpu);
  }
}

/* ------------------------------------------------------------------------- */
