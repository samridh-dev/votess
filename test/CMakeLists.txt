cmake_minimum_required(VERSION 3.18)
if (ENABLE_BUILD_TEST)

  file(GLOB SOURCE_TEST "*.cpp")
  if (NOT SOURCE_TEST)
    message(FATAL_ERROR "no source files for test executable found.")
  endif()

  add_executable(test ${SOURCE_TEST})

  target_link_libraries(test PRIVATE Catch2::Catch2 voro++)
  if (CMAKE_CXX_COMPILER MATCHES "icpx$")
    target_link_libraries(test PRIVATE sycl)
  endif()

  set_target_properties(test 
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
  )

endif()

