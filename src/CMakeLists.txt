cmake_minimum_required(VERSION 3.18)

if (ENABLE_BUILD_CLI OR ENABLE_BUILD_PYVOTESS)
  file(GLOB SOURCE_LIB "*.cpp")
  if (NOT SOURCE_LIB)
    message(FATAL_ERROR "No source files found.")
  endif()
  add_library(common STATIC ${SOURCE_LIB})

  if (CMAKE_CXX_COMPILER MATCHES "icpx$")
    target_link_libraries(common PRIVATE sycl)
  endif()
endif()

if (ENABLE_BUILD_CLI)
  add_executable(clvotess ${SOURCE_LIB})
  target_link_libraries(clvotess PRIVATE common)
  set_target_properties(clvotess 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
  )
endif()

if (ENABLE_BUILD_PYVOTESS)
  file(GLOB SOURCE_PYBIND "pybind11/*.cpp")
  if (NOT SOURCE_PYBIND)
    message(FATAL_ERROR "No source files for pyvotess Python extension found.")
  endif()
  Python_add_library(pyvotess MODULE ${SOURCE_LIB} ${SOURCE_PYBIND})
  target_link_libraries(pyvotess PRIVATE common pybind11::module)
  set_target_properties(pyvotess 
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    CXX_VISIBILITY_PRESET default 
    VISIBILITY_INLINES_HIDDEN ON
  )
endif()
