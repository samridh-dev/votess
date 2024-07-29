
# to set pthreads for linux C++
add_library(Threads::Threads INTERFACE IMPORTED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (ENABLE_BUILD_VOTESS)

endif()

if (ENABLE_BUILD_PYVOTESS)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  add_subdirectory(extern/pybind11)
endif()

if (ENABLE_BUILD_TEST)
  add_subdirectory(extern/catch2)
  add_subdirectory(extern/voropp)
endif()
