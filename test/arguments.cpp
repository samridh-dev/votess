#include <catch2/catch_test_macros.hpp>
#include "arguments.hpp"

TEST_CASE("vtargref assignment and conversion", "[vtargs]") {
  votess::vtargref obj;

  SECTION("Assigning and converting integer values") {
    obj = 53;
    REQUIRE(obj.get<std::string>() == "53");

    long int k = obj;
    REQUIRE(k == 53);

    int i = obj;
    REQUIRE(i == 53);
  }

  SECTION("Assigning and converting floating point values") {
    obj = 3.14159;
    REQUIRE(obj.get<std::string>() == "3.14159");

    double d = obj;
    REQUIRE(d == 3.14159);
  }

  SECTION("Assigning and converting string values") {
    obj = std::string("hello");
    REQUIRE(obj.get<std::string>() == "hello");

    std::string str = obj;
    REQUIRE(str == "hello");
  }

  SECTION("Assigning and converting boolean values") {
    obj = true;
    REQUIRE(obj.get<std::string>() == "1");

    bool b = obj;
    REQUIRE(b == true);

    obj = false;
    REQUIRE(obj.get<std::string>() == "0");

    b = obj;
    REQUIRE(b == false);

  }

  SECTION("Assigning and converting character values") {
    obj = 'a';
    REQUIRE(obj.get<std::string>() == "a");

    char c = obj;
    REQUIRE(c == 'a');
  }
}

TEST_CASE("vtargref type mismatch and exceptions", "[vtargs]") {
  votess::vtargref obj;

  SECTION("Invalid conversion should throw exception") {
    obj = std::string("hello");
    REQUIRE_THROWS_AS(static_cast<int>(obj), std::invalid_argument);

    obj = std::string("test");
    REQUIRE_THROWS_AS(static_cast<int>(obj), std::invalid_argument);

    obj = 'b';
    REQUIRE_THROWS_AS(static_cast<int>(obj), std::invalid_argument);
  }

  SECTION("Valid conversion should not throw exception") {
    obj = 123;
    REQUIRE_NOTHROW(static_cast<int>(obj));
    REQUIRE(static_cast<int>(obj) == 123);

    obj = std::string("world");
    REQUIRE_NOTHROW(static_cast<std::string>(obj));
    REQUIRE(static_cast<std::string>(obj) == "world");

    obj = 3.14;
    REQUIRE_NOTHROW(static_cast<double>(obj));
    REQUIRE(static_cast<double>(obj) == 3.14);

    obj = false;
    REQUIRE_NOTHROW(static_cast<bool>(obj));
    REQUIRE(static_cast<bool>(obj) == false);

    obj = true;
    REQUIRE_NOTHROW(static_cast<int>(obj));
    REQUIRE(static_cast<int>(obj) == 1);

    obj = 243;
    REQUIRE_NOTHROW(static_cast<std::string>(obj));
    REQUIRE(static_cast<std::string>(obj) == "243");

    obj = true;
    REQUIRE_NOTHROW(static_cast<bool>(obj));
    REQUIRE(static_cast<bool>(obj) == true);
  }

  SECTION("Type information should be tracked correctly") {
    obj = 42;
    REQUIRE_NOTHROW(static_cast<int>(obj));
    REQUIRE_NOTHROW(static_cast<double>(obj));

    obj = std::string("test");
    REQUIRE_NOTHROW(static_cast<std::string>(obj));
    REQUIRE_THROWS_AS(static_cast<int>(obj), std::invalid_argument);

    obj = true;
    REQUIRE_NOTHROW(static_cast<bool>(obj));
    REQUIRE_NOTHROW(static_cast<int>(obj));
  }
}

TEST_CASE("vtargs initialization and assignment", "[vtargs]") {
  votess::vtargs args;

  SECTION("Default values initialization") {
    REQUIRE(args["k"].get<int>() == ARGS_DEFAULT_K);
    REQUIRE(args["cpu_nthreads"].get<int>() == ARGS_DEFAULT_CPU_NTHREADS);
    REQUIRE(args["gpu_ndsize"].get<int>() == ARGS_DEFAULT_GPU_NDWORKSIZE);
    REQUIRE(args["chunksize"].get<int>() == ARGS_DEFAULT_CHUNKSIZE);
    REQUIRE(args["use_recompute"].get<bool>() == ARGS_DEFAULT_USE_RECOMPUTE);
    REQUIRE(args["knn_grid_resolution"].get<int>() == ARGS_DEFAULT_GRID_RESOLUTION);
    REQUIRE(args["cc_p_maxsize"].get<int>() == ARGS_DEFAULT_P_MAXSIZE);
    REQUIRE(args["cc_t_maxsize"].get<int>() == ARGS_DEFAULT_T_MAXSIZE);
  }

  SECTION("Assigning new values") {
    args["k"] = 42;
    args["cpu_nthreads"] = 8;
    args["gpu_ndsize"] = 512;
    args["chunksize"] = 2048;
    args["use_recompute"] = false;
    args["knn_grid_resolution"] = 128;
    args["cc_p_maxsize"] = 256;
    args["cc_t_maxsize"] = 512;

    REQUIRE(args["k"].get<int>() == 42);
    REQUIRE(args["cpu_nthreads"].get<int>() == 8);
    REQUIRE(args["gpu_ndsize"].get<int>() == 512);
    REQUIRE(args["chunksize"].get<int>() == 2048);
    REQUIRE(args["use_recompute"].get<bool>() == false);
    REQUIRE(args["knn_grid_resolution"].get<int>() == 128);
    REQUIRE(args["cc_p_maxsize"].get<int>() == 256);
    REQUIRE(args["cc_t_maxsize"].get<int>() == 512);
  }
}

TEST_CASE("vtargs get_xyzset", "[vtargs]") {
  votess::vtargs args;
  args["knn_grid_resolution"] = 64;

  auto xyzset = args.get_xyzset();
  REQUIRE(xyzset.grid_resolution == 64);

  // Reassign and check again
  args["knn_grid_resolution"] = 128;
  xyzset = args.get_xyzset();
  REQUIRE(xyzset.grid_resolution == 128);
}

TEST_CASE("vtargs get_knn", "[vtargs]") {
  votess::vtargs args;
  args["k"] = 42;
  args["knn_grid_resolution"] = 64;

  auto knn = args.get_knn();
  REQUIRE(knn.k == 42);
  REQUIRE(knn.grid_resolution == 64);

  // Reassign and check again
  args["k"] = 21;
  args["knn_grid_resolution"] = 128;
  knn = args.get_knn();
  REQUIRE(knn.k == 21);
  REQUIRE(knn.grid_resolution == 128);
}

TEST_CASE("vtargs get_cc", "[vtargs]") {
  votess::vtargs args;
  args["k"] = 42;
  args["cc_p_maxsize"] = 128;
  args["cc_t_maxsize"] = 256;

  auto cc = args.get_cc();
  REQUIRE(cc.k == 42);
  REQUIRE(cc.p_maxsize == 128);
  REQUIRE(cc.t_maxsize == 256);

  // Reassign and check again
  args["k"] = 84;
  args["cc_p_maxsize"] = 256;
  args["cc_t_maxsize"] = 512;
  cc = args.get_cc();
  REQUIRE(cc.k == 84);
  REQUIRE(cc.p_maxsize == 256);
  REQUIRE(cc.t_maxsize == 512);
}
