#!/bin/sh

PROJECT_DIR=$(pwd)
BUILD_DIR="${PROJECT_DIR}/cmake/build"

source /opt/intel/oneapi/setvars.sh

git submodule update --init --recursive

echo "building Makefile"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake ../.. -Wdev
echo "building Project"
make -j $(nproc)
