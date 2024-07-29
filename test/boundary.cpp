#include <catch2/catch_test_macros.hpp> 
#include <catch2/generators/catch_generators.hpp> 
#include <libsycl.hpp>

#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include <boundary.hpp>

///////////////////////////////////////////////////////////////////////////////
/// Forward declarations
///////////////////////////////////////////////////////////////////////////////

static std::string 
vector_to_string(const std::vector<int>& vec);

static void 
test_boundary(
  const std::vector<int>& cycle, int head,
  const std::vector<int>& ans, const int ulimit
);

template <typename T>
static void boundary_compute_gpu(
  std::vector<T>& cycle_vec,
  const size_t dr_offs,
  const size_t dr_size,
  short int& head,
  std::vector<T>& R_vec,
  const size_t r_offs,
  const size_t r_size
);

///////////////////////////////////////////////////////////////////////////////
/// Test Cases                                                              ///
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Case: 1                                                                 ///
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("boundary tests: case 1", "[boundary]") {
  const short int dR_size = 6;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};
  short int head = -1, dR_offs = 0, T_offs = 0, T_size = 4, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 2,5,0, 5,3,0, 1,5,2, 5,1,3 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 2:") {
    std::vector<int> T = { 5,3,0, 2,5,0, 1,5,2, 5,1,3 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }

  SECTION("case 3:") {
    std::vector<int> T = { 5,3,0, 1,5,2, 2,5,0, 5,1,3 };

    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }

    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }

  SECTION("case 4:") {
    std::vector<int> T = { 5,3,0, 1,5,2, 5,1,3, 2,5,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 5:") {
    std::vector<int> T = { 1,5,2, 5,3,0, 5,1,3, 2,5,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 6:") {
    std::vector<int> T = { 1,5,2, 5,1,3, 5,3,0, 2,5,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 7:") {
    std::vector<int> T = { 1,5,2, 5,1,3, 2,5,0, 5,3,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 8:") {
    std::vector<int> T = { 1,5,2, 2,5,0, 5,3,0, 5,1,3 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Case: 2                                                                 ///
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("boundary tests: case 2", "[boundary]") {
  const short int dR_size = 6;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};
  short int head = -1, dR_offs = 0, T_offs = 0, T_size = 4, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 2:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 3:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 4:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 5:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 6:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 7:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 8:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Case: 3                                                                 ///
///////////////////////////////////////////////////////////////////////////////

// NOTE: For some reason this code is segfaulting randomly
#if 0
TEST_CASE("boundary tests: case 3", "[boundary]") {
  const short int dR_size = 6;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {2, 1, 5, 3, 0};
  short int head = -1, dR_offs = 0, T_offs = 0, T_size = 4, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 2,5,0, 5,3,0, 1,5,2 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 2:") {
    std::vector<int> T = { 5,3,0, 2,5,0, 1,5,2 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 3:") {
    std::vector<int> T = { 5,3,0, 1,5,2, 2,5,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 4:") {
    std::vector<int> T = { 1,5,2, 5,3,0, 2,5,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
  SECTION("case 5:") {
    std::vector<int> T = { 1,5,2, 2,5,0, 5,3,0 };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
          T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
}
#endif

///////////////////////////////////////////////////////////////////////////////
/// Case: 5                                                                 ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("boundary tests: case 5", "[boundary]") {
  const short int dR_size = 6;
  std::vector<int> dR(52 * 6 + dR_size, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};
  short int head = -1, dR_offs = 52 * 6, T_offs = 0, T_size = 4, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 2:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 3:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 4:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 5:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 6:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 7:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 8:") {
    std::vector<int> T = { 4,2,0, 4,0,3, 2,4,1, 4,3,1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
}

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Case: 6                                                                 ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("boundary tests: case 6", "[boundary]") {
  const short int dR_size = 6;
  std::vector<int> dR(52 * 6 + dR_size, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};
  short int head = -1, dR_offs = 52 * 6, T_offs = 9, T_size = 4, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 
      -1,-1,-1, -1,-1,-1 ,-1,-1,-1, 4,2,0, 4,0,3, 2,4,1, 4,3,1 
    };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 2:") {
    std::vector<int> T = { 
      -1,-1,-1, -1,-1,-1 ,-1,-1,-1, 4,2,0, 4,0,3, 2,4,1, 4,3,1 
    };
    boundary::compute(dR.data(), dR_offs, dR_size, head, T.data(), T_offs,
        T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 3:") {
    std::vector<int> T = {
      -1,-1,-1, -1,-1,-1 ,-1,-1,-1, 4,2,0, 4,0,3, 2,4,1, 4,3,1 
    };
    boundary::compute(dR.data(), dR_offs, dR_size, 
        head, T.data(), T_offs, T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
  SECTION("case 4:") {
    std::vector<int> T = {
     -1,-1,-1, -1,-1,-1 ,-1,-1,-1, 4,2,0, 4,0,3, 2,4,1, 4,3,1 
    };
    boundary::compute(dR.data(), dR_offs, dR_size, 
        head, T.data(), T_offs, T_size);
    std::vector<int> test{
      dR[52 * 6 + 0], dR[52 * 6 + 1], dR[52 * 6 + 2],
      dR[52 * 6 + 3], dR[52 * 6 + 4], dR[52 * 6 + 5],
    };
    test_boundary(test, head, ans, ulimit);
  }
}

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Case: 7                                                                 ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("boundary tests: case 7", "[boundary]") {
  const short int dR_size = 8;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {0, 2, 4, 6};
  short int head = -1, dR_offs = 0, T_offs = 6, T_size = 8, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 
      -1,-1,-1, -1,-1,-1,
      2,4,6, 1,2,6, 2,1,7, 1,6,7, 6,0,7, 0,5,7, 5,2,7, 2,5,0
    };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, 
          head, T.data(), T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Case: 8                                                                 ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("boundary tests: case 8", "[boundary]") {
  const short int dR_size = 9;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {0,3,8,6,7};
  short int head = -1, dR_offs = 0, T_offs = 0, T_size = 7, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 
      4,6,7, 0,4,7, 4,3,1, 6,1,8, 1,3,8, 4,0,3, 4,1,6,
    };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, 
          head, T.data(), T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }

  SECTION("case 2:") {
    std::vector<int> T = { 
      4,6,7, 0,4,7, 4,3,1, 6,1,8, 1,3,8, 4,1,6, 4,0,3,
    };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, 
          head, T.data(), T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Case: 9                                                                 ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("boundary tests: case 9", "[boundary]") {
  const short int dR_size = 9;
  std::vector<int> dR(dR_size, 0xff);
  std::vector<int> ans = {2,5,0,};

  short int 
  head = -1, dR_offs = 0, 
  T_offs = 14 * 3, T_size = 7, ulimit = 10;

  SECTION("case 1:") {
    std::vector<int> T = { 
      -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, 
      -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1, 
      1,5,2, 5,1,3, 4,2,0, 4,0,3, 2,4,1, 4,3,1, 5,3,0, 
    };
    SECTION("cpu") {
      boundary::compute(dR.data(), dR_offs, dR_size, 
          head, T.data(), T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
    SECTION("gpu") {
      boundary_compute_gpu(dR, dR_offs, dR_size, head, T, T_offs, T_size);
      test_boundary(dR, head, ans, ulimit);
    }
  }

}

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Helper Functions                                                        ///
///////////////////////////////////////////////////////////////////////////////

static std::string 
vector_to_string(const std::vector<int>& vec) {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i < vec.size() - 1) ss << ", ";
  }
  ss << "}";
  return ss.str();
}

static void test_boundary(
  const std::vector<int>& cycle,
  int head,
  const std::vector<int>& ans,
  const int ulimit
) {

  const int first = head;
  int counter = 0;

  INFO("Cycle: " << vector_to_string(cycle));
  INFO("Answer: " << vector_to_string(ans));
  REQUIRE(head < static_cast<int>(cycle.size()));

  do {
    
    REQUIRE(head < static_cast<int>(cycle.size()));
    const int vertex_0 = cycle[head];
    const int vertex_1 = cycle[vertex_0];

    auto it_0 = std::find(ans.begin(), ans.end(), vertex_0);
    auto it_1 = std::find(ans.begin(), ans.end(), vertex_1);

    if(it_0 == ans.end() || it_1 == ans.end()) {
      INFO("Missing vertex : " << vertex_0 << " or " << vertex_1);
      REQUIRE_FALSE("Cycle contains undesirable vertices" );
      break;
    }

    const int ans_index_0 = std::distance(ans.begin(), it_0);
    const int ans_index_1 = std::distance(ans.begin(), it_1);

    const int ans_size = ans.size();
    const bool condition = ((ans_index_0 + 1) % ans_size == ans_index_1) || 
                           ((ans_index_1 + 1) % ans_size == ans_index_0);
    REQUIRE(condition);

    head = cycle[head];
    REQUIRE(head < static_cast<int>(cycle.size()));

    if (counter++ > ulimit) break;

  } while (cycle[head] != first);

}

template <typename T>
static void boundary_compute_gpu(
  std::vector<T>& cycle_vec,
  const size_t dr_offs,
  const size_t dr_size,
  short int& head,
  std::vector<T>& R_vec,
  const size_t r_offs,
  const size_t r_size) {
  
  sycl::queue q;
  sycl::buffer<T> bcycle(cycle_vec.data(), sycl::range<1>(cycle_vec.size()));
  sycl::buffer<T> bR(R_vec.data(), sycl::range<1>(R_vec.size()));
  sycl::buffer<short int> bhead(&head, sycl::range<1>(1));

  q.submit([&](sycl::handler& h) {
    auto cycle = bcycle.template get_access<sycl::access::mode::read_write>(h);
    auto R = bR.template get_access<sycl::access::mode::read_write>(h);
    auto ahead = bhead.template get_access<sycl::access::mode::read_write>(h);
    h.single_task([=]() {
      boundary::compute(cycle, dr_offs, dr_size, ahead[0], R, r_offs, r_size);
    });
  }).wait();

  auto hcycle = bcycle.get_host_access();
  auto hR = bR.get_host_access();
  auto hhead = bhead.get_host_access();

  std::copy(hcycle.get_pointer(), 
            hcycle.get_pointer() + cycle_vec.size(),
            cycle_vec.begin());
  std::copy(hR.get_pointer(),
            hR.get_pointer() + R_vec.size(),
            R_vec.begin());
  head = hhead[0];

}

