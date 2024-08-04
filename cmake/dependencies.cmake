
# to set pthreads for linux C++
add_library(Threads::Threads INTERFACE IMPORTED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (ENABLE_BUILD_CLI)

endif()

if (ENABLE_BUILD_PYVOTESS)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  add_subdirectory(extern/pybind11)
endif()

if (ENABLE_BUILD_TEST)

  add_subdirectory(extern/catch2)

  set(VORO_BUILD_SHARED_LIBS OFF)
  set(VORO_BUILD_EXAMPLES OFF)
  set(VORO_BUILD_CMD_LINE OFF)
  set(VORO_ENABLE_DOXYGEN OFF)
  add_subdirectory(extern/voropp)

endif()
