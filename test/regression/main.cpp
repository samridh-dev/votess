#include <vector>
#include <array>
#include <chrono>
#include <getopt.h>
#include <iostream>

#include <votess.hpp>

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

#include <voro++.hh>
template <typename T>
static std::pair<std::vector<std::array<T, 3>>, std::vector<std::vector<int>>>
run_voro(
  const std::vector<std::array<T,3>>& xyzset,
  const class votess::vtargs vtargs
) {
  std::vector<std::array<T, 3>> coords;
  std::vector<std::vector<int>> neighbor_list;
  const double tolerance = 1e-8;

  using namespace voro;
  container con(
    0, 1, 0, 1, 0, 1,
    vtargs.get_xyzset().grid_resolution,
    vtargs.get_xyzset().grid_resolution,
    vtargs.get_xyzset().grid_resolution,
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

template <typename T>
static int run_test(
  std::vector<std::array<T,3>>& xyzset,
  class votess::vtargs vtargs,
  const enum votess::device device
) {
  // __internal__suppress_stdout s; // to preventstdout 

  (void)xyzset::sort<int,T>(xyzset, vtargs.get_xyzset());

  auto [vcoord, vneighbor] = run_voro<T>(xyzset, vtargs);
  auto dnn = votess::tesellate<int, T>(xyzset, vtargs, device);

  std::vector<int> test_dnn(0);
  std::vector<int> test_vneighbor(0);

  int nerrors = 0;

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
      std::cerr << "Error: Matching coordinate not found in vcoord\n";
      continue;
    }

    const size_t index = std::distance(vcoord.begin(), it);

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

    bool cond = std::includes(
      test_dnn.begin(),
      test_dnn.end(),
      test_vneighbor.begin(),
      test_vneighbor.end()
    );
    
    if (!cond) {

      std::cerr << "Error: [ i : " << i << ", test_dnn: ";
      for (const auto& val : test_dnn) {
        std::cerr << val << " ";
      }
      std::cerr << ", test_vneighbor: ";
      for (const auto& val : test_vneighbor) {
        std::cerr << val << " ";
      }
      std::cerr << "]\n";
      nerrors++;

    }

  }
  
  return nerrors;

}

#include <random>
static std::vector<std::array<float, 3>> generate_set(int count) {
  std::vector<std::array<float, 3>> xyzset;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.001f, 0.999);
  for (size_t i = 0; i < count; ++i) {
   xyzset.push_back({dis(gen), dis(gen), dis(gen)});
  }
  return xyzset;
}

static std::vector<std::array<float, 3>> read_set(const std::string& fname) {
  std::ifstream infile(fname);
  std::string line;
  std::vector<std::array<float, 3>> xyzset;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::array<float, 3> point;
    if (!(iss >> point[0] >> point[1] >> point[2])) {
      break;
    }
    xyzset.push_back(point);
  }
  return xyzset;
}

static void save_set(const std::vector<std::array<float, 3>>& xyzset,
                     const std::string& fname) {

  std::ofstream outfile(fname);
  if (outfile.is_open()) {
    for (const auto& coords : xyzset) {
      outfile << coords[0] << " " << coords[1] << " " << coords[2] << "\n";
    }
    outfile.close();
  } else {
    std::cerr << "Error: Could not open file to save dataset\n";
  }

}

int main(int argc, char* argv[]) {
  int k = 80;
  int gr = 16;
  int N = 100000;
  std::string fname;

  int option;
  while ((option = getopt(argc, argv, "k:g:N:f:")) != -1) {
    switch (option) {
      case 'k': { k = std::atoi(optarg); break; }
      case 'g': { gr = std::atoi(optarg); break; }
      case 'N': { N = std::atoi(optarg); break; }
      case 'f': { fname = optarg; break; }
      default:
        std::cerr << "Usage: " << argv[0]
                  << " [-k value] [-g value] [-N value] [-f fname]"
                  << std::endl;
        return 1;
    }
  }

  k = k < N ? k : N - 1;
  std::cout << "k : " << k << std::endl;

  std::vector<std::array<float, 3>> xyzset = fname.empty() ? 
                                              generate_set(N):
                                              read_set(fname);
  class votess::vtargs args;
  args["k"] = k;
  args["knn_grid_resolution"] = gr;
  args["gpu_ndsize"] = 32;

  save_set(xyzset, "dat/fail.xyz");

  auto start = std::chrono::high_resolution_clock::now();
  int nerrors = run_test(xyzset, args, votess::device::cpu);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  std::cout << "CPU Execution time: " << elapsed.count() << " seconds" 
            << ", Number of errors reported: " << nerrors
            << std::endl;

  // Use this when you have a working gpu implementation.
  args["use_chunking"] = false;
  start = std::chrono::high_resolution_clock::now();
  nerrors = run_test(xyzset, args, votess::device::gpu);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "GPU Execution time: " << elapsed.count() << " seconds" 
            << ", Number of errors reported: " << nerrors
            << std::endl;

  return 0;

}

