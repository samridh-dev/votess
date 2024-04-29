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

"Options:\n"
" -h, --help              Show help\n"
" -v, --version           Show version\n"
" -i, --infile  <infile>  Specify input file. Required argument.\n"
" -k, --k_init  <k_init>  Specify initial k for k-nearest neighbor search.\n"

  << std::endl;
}

static void
print_help_all(const char* const executable) {
  (void)executable;
  std::cout <<
    
"Options:\n"

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

int main(int argc, char* argv[]) {

  std::unordered_map<std::string, std::function<void(int &)>> args;

  if (argc == 1) {
    print_usage(argv[0]);
    return 0;
  }
  args["-v"] = args["--version"] = [&](int& i) {
    (void)i;
    print_version(argv[0]);
    return 0;
  };
  args["-h"] = args["--help"] = [&](int& i) {
    (void)i;
    print_help(argv[0]);
    return 0;
  };
  args["--help-all"] = [&](int& i) {
    (void)i;
    print_help_all(argv[0]);
    return 0;
  };

  args["-i"] = args["--infile"] = [&](int& i) {
    infile = argv[++i];
  };
  args["-k"] = args["--k-init"] = [&](int& i) {
    k_init = std::stoi(argv[++i]);
  };
  args["--grid-resolution"] = [&](int& i) {
    grid_resolution = std::stoi(argv[++i]);
  };
  
  for (int i = 1; i < argc; ++i) {
  auto it = args.find(argv[i]);
    if (it != args.end()) {
      it->second(i);
    } else {
      std::cerr << "Error: unknown or incomplete argument: " 
                << argv[i] 
                << std::endl;
      return 1;
    }
  }
 
  std::vector<std::array<float, 3>> xyzset;
  {
    std::ifstream file(infile);
    std::string line;

    if (!file) {
      std::cerr << "Error: Could not open file " << infile << std::endl;
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
  }
  
  if (k_init == 0) {
    k_init = xyzset.size() < default_k_init ? xyzset.size() : default_k_init;
  }

  struct votess::vtargs vtargs(k_init, grid_resolution);
  auto dnn = votess::tesellate<int, float>(xyzset, vtargs);

  return 0;
}
