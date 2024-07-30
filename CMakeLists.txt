cmake_minimum_required(VERSION 3.18)

include(cmake/unset_opts.cmake)
include(cmake/options.cmake)
include(cmake/compiler_settings.cmake)

project(votess
  VERSION 0.1
  DESCRIPTION "a parallel multi-target voronoi tesselator"
  LANGUAGES CXX
)

include(cmake/dependencies.cmake)
include(cmake/include.cmake)

add_subdirectory(src)
add_subdirectory(test)

# Add votess as a library
add_library(libvotess INTERFACE)
target_include_directories(libvotess INTERFACE 
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
