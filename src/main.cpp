#include <votess.hpp>
#include <libsycl.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <functional>

///////////////////////////////////////////////////////////////////////////////
/// Helper Functions
///////////////////////////////////////////////////////////////////////////////

static void
print_usage(const char* const executable) {
  std::cout <<
  "Usage: " << executable << " -i <input>"
  << std::endl;
}


static void
print_version(const char* const executable) {
  (void)executable;
}

static void
print_help(const char* const executable) {
  (void)executable;
  std::cout <<

"\nOptions:\n"
" -h, --help               Show help\n"
" -v, --version            Show version\n"
" -i, --infile   <infile>  Specify input file. Required argument.\n"
" -k, --k-init   <k_init>  Specify initial k for k-nearest neighbor search.\n"
" -g, --gridres  <k_init>  Specify grid resolution.\n"

  << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/// Command line Parameters
///////////////////////////////////////////////////////////////////////////////

static const int default_k_init = 64;

static std::string infile = "";
static int k_init = 0;
static int grid_resolution = 24;
  
///////////////////////////////////////////////////////////////////////////////
/// Main
///////////////////////////////////////////////////////////////////////////////

#include <getopt.h>
#if 0
int main(int argc, char* argv[]) {

  int opt = 0;
  int option_index = 0;

  struct option long_options[] = {
    {"version",           no_argument,        0,  'v'},
    {"help",              no_argument,        0,  'h'},
    {"infile",            required_argument,  0,  'i'},
    {"k-init",            required_argument,  0,  'k'},
    {"grid-resolution",   required_argument,  0,  'g'},
    {0, 0, 0, 0}
  };

  while (1) {

    opt = getopt_long(argc, argv, "vhi:k:g:", long_options, &option_index);
    if (opt == -1) break;

    switch (opt) {
      case 'v':
        print_version(argv[0]);
        return 0;
      case 'h':
        print_help(argv[0]);
        return 0;
      case 'i':
        infile = optarg;
        break;
      case 'k':
        k_init = std::atoi(optarg);
        break;
      case 'g':
        grid_resolution = std::atoi(optarg);
        break;
      case '?': 
        return 1;
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  if (infile == "") {
    std::cerr << "Error: Input file must be specified." << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  std::vector<std::array<float, 3>> xyzset;

  std::ifstream file(infile);
  std::string line;

  if (!file) {
    std::cerr << "Error: Could not open file: " << infile << std::endl;
    return 1;
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::array<float, 3> point;
    if (!(iss >> point[0] >> point[1] >> point[2])) {
      std::cerr << "Error: Incorrect data format in file" << std::endl;
      return 1;
    }
    xyzset.push_back(point);
  }
  
  if (k_init == 0) {
    k_init = xyzset.size() < default_k_init ? 
             xyzset.size() : default_k_init;
  }

  class votess::vtargs vtargs;
  vtargs["k"] = k_init;
  vtargs["knn_grid_resolution"] = grid_resolution;
  auto dnn = votess::tesellate<int, float>(xyzset, vtargs);

  return 0;
}
#else

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

int main(int argc, char* argv[]) {

  int k = 64;
  int gr = 24;
  int N = 100000;

  int option;
  while ((option = getopt(argc, argv, "k:g:N:")) != -1) {
    switch (option) {
      case 'k': { k = std::atoi(optarg); break; }
      case 'g': { gr = std::atoi(optarg); break; }
      case 'N': { N = std::atoi(optarg); break; }
      default:
        std::cerr << "Usage: " << argv[0] 
                  << " [-k value] [-g value] [-N value]" 
                  << std::endl;
        return 1;
    }
  }
  
  auto xyzset = generate_set(N);

  class votess::vtargs args;
  args["k"] = k;
  args["knn_grid_resolution"] = gr;
  args["gpu_ndsize"] = 24;
  args["use_recompute"] = false;
  args["dev_suppress_stdout"] = false;

  args["use_chunking"] = false;
  args["chunksize"] = 102400;

#if 1
  
  {
    args["cc_p_maxsize"] = 32;
    args["cc_t_maxsize"] = 64;
    auto start = std::chrono::high_resolution_clock::now();
    auto dnn = votess::tesellate<int32_t, float>(xyzset, args, 
                                                 votess::device::gpu);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    std::cout << "[GPU] Execution time: " 
              << elapsed.count() << " seconds" 
              << std::endl;
  }

#endif

#if 1
  
  {
    args["use_chunking"] = true;
    args["chunksize"] = 8196;
    args["cc_p_maxsize"] = 32;
    args["cc_t_maxsize"] = 64;

    auto start = std::chrono::high_resolution_clock::now();
    auto dnn = votess::tesellate<int, float>(xyzset, args,
                                             votess::device::cpu);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    std::cout << "[CPU] Execution time: " 
              << elapsed.count() << " seconds" 
              << std::endl;
  }

#endif

  return 0;

}

#endif
