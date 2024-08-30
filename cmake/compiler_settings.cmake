set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
else()
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ADDITIONAL_DEBUG_FLAGS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(ADDITIONAL_DEBUG_FLAGS 
    "${ADDITIONAL_DEBUG_FLAGS} -Wduplicated-cond                              \
    -Wduplicated-branches -Wlogical-op -Wuseless-cast"
  )
endif()

set(CMAKE_CXX_FLAGS_ALL 
  ""
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}                           \
   -Wall -Wextra -Wshadow -Wpedantic -Wformat=2 -fno-omit-frame-pointer -O0"
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}                       \
  -O3 -funroll-loops"
)

set(CMAKE_CXX_LINK_FLAGS_ALL 
  "-pthread"
)

# compiler specifics
if (CMAKE_CXX_COMPILER MATCHES "icpx$")

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_ALL}                           \
                             ${CMAKE_CXX_FLAGS_DEBUG}                         \
                             ${CMAKE_CXX_LINK_FLAGS_ALL}                      \
                             -fsycl")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_ALL}                         \
                               ${CMAKE_CXX_FLAGS_RELEASE}                     \
                               ${CMAKE_CXX_LINK_FLAGS_ALL}                    \
                               -g3                                            \
                               -gline-tables-only                             \
                               -fdebug-info-for-profiling                     \
                               -debug inline-debug-info                       \
                               -fsycl")

endif()

if (CMAKE_CXX_COMPILER MATCHES "acpp$")

  find_package(AdaptiveCpp REQUIRED)

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_ALL}                           \
                             ${CMAKE_CXX_FLAGS_DEBUG}                         \
                             ${CMAKE_CXX_LINK_FLAGS_ALL}                      \
                             -fPIC                                            \
                             ")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_ALL}                         \
                               ${CMAKE_CXX_FLAGS_RELEASE}                     \
                               ${CMAKE_CXX_LINK_FLAGS_ALL}                    \
                               -fPIC                                          \
                               ")

endif()

set(SYCL_LIBRARY "<CL/sycl.hpp>")
if(CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(SYCL_LIBRARY "<CL/sycl.hpp>")
elseif(CMAKE_CXX_COMPILER MATCHES "acpp$")
  set(SYCL_LIBRARY "<sycl/sycl.hpp>")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libsycl.hpp.in 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/libsycl.hpp
)
