#!/bin/sh

CXX_COMPILER=icpx
BUILD_DIR="cmake/build"
INTEL_ENV="/opt/intel/oneapi/setvars.sh"

show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -c, --compiler   Set compiler or compiler path"
  echo "  -h, --help       Display this help message"
}

# Parse command line arguments using getopt
OPTIONS=$(getopt -o c:h --long compiler:,help -- "$@")
if [ $? -ne 0 ]; then
  show_help
  exit 1
fi
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -c|--compiler)
      CXX_COMPILER="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

if [[ "$CXX_COMPILER" == *icpx* || "$CXX_COMPILER" == *dpcpp* ]]; then
  if [ -f "$INTEL_ENV" ]; then
    source "$INTEL_ENV"
  else
    echo "Error: Intel environment file $INTEL_ENV not found"
    exit 1
  fi
fi

# Build Votess
PROJECT_ROOT=$(pwd)
git submodule update --init --recursive
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake ${PROJECT_ROOT} -Wdev -DCMAKE_CXX_COMPILER=$(which $CXX_COMPILER)
make -j $(nproc)
