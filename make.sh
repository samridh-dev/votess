#!/bin/sh

PROJECT_DIR=$(pwd)
BUILD_DIR="${PROJECT_DIR}/cmake/build"

# Source oneapi/setvars.sh for compilation
INTEL_ENV="/opt/intel/oneapi/setvars.sh"
if [ -f "$INTEL_ENV" ]; then
  echo "Sourcing Intel oneAPI environment..."
  source "$INTEL_ENV"
else
  echo "file ${INTEL_ENV} not found"
fi

echo "Acquiring dependencies"
git submodule update --init --recursive

echo "building Makefile"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake ../.. -Wdev
echo "building Project"
make -j $(nproc)
