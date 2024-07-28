#include <catch2/catch_session.hpp>
#include <iostream>
#include <string>
#include <libsycl.hpp>
#include <catch2/catch_test_macros.hpp> 

int main(int argc, char *args[]) {

  {
    sycl::queue q;
    auto device = q.get_device();
    if (!device.has(sycl::aspect::fp64)) {

      std::cerr << "\e[93mWarning: \e[00m" 
                << "Device does not support double precision (fp64). "
                << "Skipping all device fp64 tests.\n";

    }
  }

  return Catch::Session().run(argc, args);

}
