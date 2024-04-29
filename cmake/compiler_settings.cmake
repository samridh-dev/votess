set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (USE_ACPP)
  set(CMAKE_CXX_COMPILER acpp CACHE PATH "C++ compiler")
else()
  set(CMAKE_CXX_COMPILER icpx CACHE PATH "C++ compiler")
endif()

if (ENABLE_DEBUG)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
else()
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Initialize variable for additional debug flags
set(ADDITIONAL_DEBUG_FLAGS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(ADDITIONAL_DEBUG_FLAGS 
    "${ADDITIONAL_DEBUG_FLAGS} -Wduplicated-cond \
    -Wduplicated-branches -Wlogical-op -Wuseless-cast"
  )
endif()

# Global
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
   -g -Wall -Wextra -Wshadow -Wpedantic -Wformat=2 -fno-omit-frame-pointer -O0"
)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
  -O3"
)

# compiler specifics
if (CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsycl")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl")
endif()

if (CMAKE_CXX_COMPILER MATCHES "acpp$")
  find_package(AdaptiveCpp REQUIRED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(SYCL_LIBRARY "<CL/sycl.hpp>")
if(CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(SYCL_LIBRARY "<CL/sycl.hpp>")
elseif(CMAKE_CXX_COMPILER MATCHES "acpp$")
  set(SYCL_LIBRARY "<sycl/sycl.hpp>")
endif()
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/libsycl.hpp.in 
  ${CMAKE_SOURCE_DIR}/include/libsycl.hpp
)
