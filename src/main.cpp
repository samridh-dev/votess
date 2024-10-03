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
#include <cstdlib>

#include <getopt.h>

///////////////////////////////////////////////////////////////////////////////
/// Helper Functions
///////////////////////////////////////////////////////////////////////////////

#if 1

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

static std::tuple <std::string, struct votess::vtargs, enum votess::device>
parse_args(int argc, char* argv[]) {

  int opt = 0;
  int option_index = 0;

  std::string infile = "";
  struct votess::vtargs vtargs;
  votess::device device = votess::device::gpu;

  int k_init = ARGS_DEFAULT_K;
  int chunksize = ARGS_DEFAULT_CHUNKSIZE;
  int cc_p_maxsize = ARGS_DEFAULT_P_MAXSIZE;
  int cc_t_maxsize = ARGS_DEFAULT_T_MAXSIZE;
  int cpu_nthreads = ARGS_DEFAULT_CPU_NTHREADS;
  int gpu_ndsize = ARGS_DEFAULT_GPU_NDWORKSIZE;
  bool use_chunking = ARGS_DEFAULT_USE_CHUNKING;
  bool use_recompute = ARGS_DEFAULT_USE_RECOMPUTE;
  int grid_resolution = ARGS_DEFAULT_GRID_RESOLUTION;

  struct option long_options[] = {
    {"version",           no_argument,        0,  'v'},
    {"help",              no_argument,        0,  'h'},
    {"infile",            required_argument,  0,  'i'},
    {"use-device",        required_argument,  0,  'x'},
    {"k-init",            required_argument,  0,  'k'},
    {"grid-resolution",   required_argument,  0,  'g'},
    {"cpu-nthreads",      required_argument,  0,  't'},
    {"gpu-ndsize",        required_argument,  0,  'd'},
    {"chunksize",         required_argument,  0,  'c'},
    {"use-chunking",      no_argument,        0,  'u'},
    {"use-recompute",     no_argument,        0,  'r'},
    {"p-maxsize",         required_argument,  0,  'p'},
    {"t-maxsize",         required_argument,  0,  'm'},
    {0, 0, 0, 0}
  };


  while ((opt = getopt_long(argc, (char* const*)argv, 
          "vhi:x:k:g:t:d:c:urp:m:", long_options, &option_index)) != -1) {

    switch (opt) {
      case 'v':
        std::cout << "Version: 1.0" << std::endl;
        break;
      case 'h':
        std::cout << "Help message." << std::endl;
        break;
      case 'i':
        infile = optarg;
        break;
      case 'x':
        if (strcmp(optarg, "cpu") == 0)      device = votess::device::cpu;
        else if (strcmp(optarg, "gpu") == 0) device = votess::device::gpu;
        else {
          std::cerr << "Error: " 
                    << "Unknown device type. Use 'cpu' or 'gpu'." 
                    << std::endl;
        }
        break;
      case 'k':
        k_init = std::atoi(optarg);
        vtargs["k"] = k_init;
        break;
      case 'g':
        grid_resolution = std::atoi(optarg);
        vtargs["knn_grid_resolution"] = grid_resolution;
        break;
      case 't':
        cpu_nthreads = std::atoi(optarg);
        vtargs["cpu_nthreads"] = cpu_nthreads;
        break;
      case 'd':
        gpu_ndsize = std::atoi(optarg);
        vtargs["gpu_ndsize"] = gpu_ndsize;
        break;
      case 'c':
        chunksize = std::atoi(optarg);
        vtargs["chunksize"] = chunksize;
        break;
      case 'u':
        use_chunking = true;
        vtargs["use_chunking"] = use_chunking;
        break;
      case 'r':
        use_recompute = true;
        vtargs["use_recompute"] = use_recompute;
        break;
      case 'p':
        cc_p_maxsize = std::atoi(optarg);
        vtargs["cc_p_maxsize"] = cc_p_maxsize;
        break;
      case 'm':
        cc_t_maxsize = std::atoi(optarg);
        vtargs["cc_t_maxsize"] = cc_t_maxsize;
        break;
      case '?':
        std::cerr << "Unknown option" << std::endl;
        break;
      default:
        std::cerr << "Usage error" << std::endl;
        break;
    }
  }

  return std::make_tuple(infile, vtargs, device);

}

int 
main(int argc, char* argv[]) {

  auto [infile, vtargs, device] = parse_args(argc, argv); 

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
  
  if (vtargs["k"].get<int>() == 0) {

    vtargs["k"] = xyzset.size() < default_k_init ? 
                  xyzset.size() : default_k_init;

  }

  auto dnn = votess::tesellate<int, float>(xyzset, vtargs);

  dnn.print();

  return 0;
}

#else

#include <random>
static std::vector<std::array<float, 3>> generate_set(int count) {
  std::vector<std::array<float, 3>> xyzset;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.001f, 0.999);

  for (int i = 0; i < count; ++i) {
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
  args["gpu_ndsize"] = 32;
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
