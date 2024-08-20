#include <catch2/catch_test_macros.hpp> 
#include <catch2/generators/catch_generators.hpp> 
#include <libsycl.hpp>
#include <boundary.hpp>

#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

namespace boundary {

template <typename T>
static void compute_gpu(
  std::vector<T>& cycle,
  const size_t dr_offs,
  const size_t dr_size,
  short int& head,
  std::vector<T>& R,
  const size_t r_offs,
  const size_t r_size) {
  
  sycl::queue q;
  sycl::buffer<T> bcycle(cycle.data(), sycl::range<1>(cycle.size()));
  sycl::buffer<T> bR(R.data(), sycl::range<1>(R.size()));
  sycl::buffer<short int> bhead(&head, sycl::range<1>(1));

  q.submit([&](sycl::handler& h) {
    auto cycle = bcycle.template 
    get_access<sycl::access::mode::read_write>(h);
    auto R = bR.template 
    get_access<sycl::access::mode::read_write>(h);
    auto ahead = bhead.template 
    get_access<sycl::access::mode::read_write>(h);
    sycl::local_accessor<T> 
    lcycle(sycl::range<1>(cycle.size()), h);
    
    sycl::nd_range<1> ndRange(1, 1);
    h.parallel_for(ndRange, [=](sycl::nd_item<1> item) {

      for (size_t i = 0; i < cycle.size(); i++) {
        lcycle[i] = cycle[i];
      }

      boundary::compute(lcycle, dr_offs, dr_size, 
                        ahead[0], R, r_offs, r_size);

      for (size_t i = 0; i < cycle.size(); i++) {
        cycle[i] = lcycle[i];
      }

    });
  }).wait();

  auto hcycle = bcycle.get_host_access();
  auto hR = bR.get_host_access();
  auto hhead = bhead.get_host_access();

  std::copy(hcycle.get_pointer(), 
            hcycle.get_pointer() + cycle.size(),
            cycle.begin());
  std::copy(hR.get_pointer(), hR.get_pointer() + R.size(),
            R.begin());
  head = hhead[0];

}

} // namespace boundary

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

namespace boundary {

static void test(
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
    const bool b = ((ans_index_0 + 1) % ans_size == ans_index_1)||
                   ((ans_index_1 + 1) % ans_size == ans_index_0);
    REQUIRE(b);

    head = cycle[head];
    REQUIRE(head < static_cast<int>(cycle.size()));

    if (counter++ > ulimit) break;

  } while (cycle[head] != first);

}

} // namespace boundary
TEST_CASE("[CPU] boundary tests: case 1", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 5, 0], [5, 3, 0], [1, 5, 2], [5, 1, 3]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 1, 5, 2, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 5, 0], [5, 3, 0], [5, 1, 3], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 5, 1, 3, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[2, 5, 0], [1, 5, 2], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[2, 5, 0], [1, 5, 2], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[2, 5, 0], [5, 1, 3], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 1, 3, 5, 3, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[2, 5, 0], [5, 1, 3], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 5, 1, 3, 1, 5, 2, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[5, 3, 0], [2, 5, 0], [1, 5, 2], [5, 1, 3]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 1, 5, 2, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[5, 3, 0], [2, 5, 0], [5, 1, 3], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 5, 1, 3, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[5, 3, 0], [1, 5, 2], [2, 5, 0], [5, 1, 3]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 2, 5, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[5, 3, 0], [1, 5, 2], [5, 1, 3], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 5, 1, 3, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[5, 3, 0], [5, 1, 3], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 5, 1, 3, 2, 5, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[5, 3, 0], [5, 1, 3], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 5, 1, 3, 1, 5, 2, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[1, 5, 2], [2, 5, 0], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[1, 5, 2], [2, 5, 0], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[1, 5, 2], [5, 3, 0], [2, 5, 0], [5, 1, 3]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 2, 5, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[1, 5, 2], [5, 3, 0], [5, 1, 3], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 5, 1, 3, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[1, 5, 2], [5, 1, 3], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 1, 3, 2, 5, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[5, 1, 3], [2, 5, 0], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 1, 3, 2, 5, 0, 5, 3, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[5, 1, 3], [2, 5, 0], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 5, 1, 3, 2, 5, 0, 1, 5, 2, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[5, 1, 3], [5, 3, 0], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 1, 3, 5, 3, 0, 2, 5, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[5, 1, 3], [5, 3, 0], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 1, 3, 5, 3, 0, 1, 5, 2, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[5, 1, 3], [1, 5, 2], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 5, 1, 3, 1, 5, 2, 2, 5, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[5, 1, 3], [1, 5, 2], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 5, 1, 3, 1, 5, 2, 5, 3, 0, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 1", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 5, 0], [5, 3, 0], [1, 5, 2], [5, 1, 3]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 1, 5, 2, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 5, 0], [5, 3, 0], [5, 1, 3], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 5, 1, 3, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[2, 5, 0], [1, 5, 2], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[2, 5, 0], [1, 5, 2], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[2, 5, 0], [5, 1, 3], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 1, 3, 5, 3, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[2, 5, 0], [5, 1, 3], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 5, 1, 3, 1, 5, 2, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[5, 3, 0], [2, 5, 0], [1, 5, 2], [5, 1, 3]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 1, 5, 2, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[5, 3, 0], [2, 5, 0], [5, 1, 3], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 5, 1, 3, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[5, 3, 0], [1, 5, 2], [2, 5, 0], [5, 1, 3]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 2, 5, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[5, 3, 0], [1, 5, 2], [5, 1, 3], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 5, 1, 3, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[5, 3, 0], [5, 1, 3], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 5, 1, 3, 2, 5, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[5, 3, 0], [5, 1, 3], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 5, 1, 3, 1, 5, 2, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[1, 5, 2], [2, 5, 0], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[1, 5, 2], [2, 5, 0], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[1, 5, 2], [5, 3, 0], [2, 5, 0], [5, 1, 3]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 2, 5, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[1, 5, 2], [5, 3, 0], [5, 1, 3], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 5, 1, 3, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[1, 5, 2], [5, 1, 3], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 1, 3, 2, 5, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[5, 1, 3], [2, 5, 0], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 1, 3, 2, 5, 0, 5, 3, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[5, 1, 3], [2, 5, 0], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 5, 1, 3, 2, 5, 0, 1, 5, 2, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[5, 1, 3], [5, 3, 0], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 1, 3, 5, 3, 0, 2, 5, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[5, 1, 3], [5, 3, 0], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 1, 3, 5, 3, 0, 1, 5, 2, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[5, 1, 3], [1, 5, 2], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 5, 1, 3, 1, 5, 2, 2, 5, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[5, 1, 3], [1, 5, 2], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 5, 1, 3, 1, 5, 2, 5, 3, 0, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
