cmake_minimum_required(VERSION 3.18)
if (ENABLE_BUILD_REGR)

  file(GLOB SOURCE_REGRESSION "*.cpp")
  if (NOT SOURCE_REGRESSION)
    message(FATAL_ERROR "no source files for regression executable found.")
  endif()

  add_executable(regression ${SOURCE_REGRESSION})

  target_link_libraries(regression PRIVATE Catch2::Catch2 voro++)
  if (CMAKE_CXX_COMPILER MATCHES "icpx$")
    target_link_libraries(regression PRIVATE sycl)
  endif()

  set_target_properties(regression 
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
  )

endif()

