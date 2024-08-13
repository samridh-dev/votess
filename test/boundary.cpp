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
  sycl::buffer<T>
  bcycle(cycle.data(), sycl::range<1>(cycle.size()));
  sycl::buffer<T> bR(R.data(), sycl::range<1>(R.size()));
  sycl::buffer<short int> bhead(&head, sycl::range<1>(1));

  q.submit([&](sycl::handler& h) {
    auto cycle = bcycle.template
    get_access<sycl::access::mode::read_write>(h);
    auto R = bR.template
    get_access<sycl::access::mode::read_write>(h);
    auto ahead = bhead.template
    get_access<sycl::access::mode::read_write>(h);
    h.single_task([=]() {
      boundary::compute(cycle, dr_offs, dr_size,
                        ahead[0], R, r_offs, r_size);
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
TEST_CASE("[CPU] boundary tests: case 2", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 2", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 3", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 3", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 4", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {2, 1, 5, 3, 0};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 3;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 5, 0], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 5, 0], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[5, 3, 0], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 1, 5, 2 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[5, 3, 0], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[1, 5, 2], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[1, 5, 2], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 4", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {2, 1, 5, 3, 0};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 3;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 5, 0], [5, 3, 0], [1, 5, 2]}") {
    std::vector<int> T = { 2, 5, 0, 5, 3, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 5, 0], [1, 5, 2], [5, 3, 0]}") {
    std::vector<int> T = { 2, 5, 0, 1, 5, 2, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[5, 3, 0], [2, 5, 0], [1, 5, 2]}") {
    std::vector<int> T = { 5, 3, 0, 2, 5, 0, 1, 5, 2 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[5, 3, 0], [1, 5, 2], [2, 5, 0]}") {
    std::vector<int> T = { 5, 3, 0, 1, 5, 2, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[1, 5, 2], [2, 5, 0], [5, 3, 0]}") {
    std::vector<int> T = { 1, 5, 2, 2, 5, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[1, 5, 2], [5, 3, 0], [2, 5, 0]}") {
    std::vector<int> T = { 1, 5, 2, 5, 3, 0, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 5", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(384, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 368;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 5", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(384, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 368;
  short int T_offs = 0;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 6", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(384, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 368;
  short int T_offs = 9;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 6", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(384, 0xff);
  std::vector<int> ans = {0, 2, 1, 3};

  short int head = -1;
  short int dR_offs = 368;
  short int T_offs = 9;
  short int T_size = 4;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 7", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(192, 0xff);
  std::vector<int> ans = {2, 5, 0};

  short int head = -1;
  short int dR_offs = 176;
  short int T_offs = 15;
  short int T_size = 7;
  short int ulimit = 10;

  SECTION("permutation 1: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 7", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(192, 0xff);
  std::vector<int> ans = {2, 5, 0};

  short int head = -1;
  short int dR_offs = 176;
  short int T_offs = 15;
  short int T_size = 7;
  short int ulimit = 10;

  SECTION("permutation 1: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[1, 5, 2], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[1, 5, 2], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[1, 5, 2], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[1, 5, 2], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 2, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 2, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 2, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 2, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [4, 2, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 4, 2, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 2, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 2, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[1, 5, 2], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[1, 5, 2], [4, 2, 0], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 1, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 1, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [4, 3, 1], [5, 3, 0], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 4, 3, 1, 5, 3, 0, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[1, 5, 2], [4, 2, 0], [4, 0, 3], [5, 3, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 0, 3, 5, 3, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 0, 3], [5, 3, 0], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 0, 3, 5, 3, 0, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[1, 5, 2], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 1, 3], [5, 3, 0], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 1, 3, 5, 3, 0, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [4, 0, 3], [5, 3, 0], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 4, 0, 3, 5, 3, 0, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [4, 0, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 4, 0, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [5, 3, 0], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 5, 3, 0, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 1, 3], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 1, 3, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 3, 0], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 3, 0, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[1, 5, 2], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 0, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 0, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 0, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 0, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 3, 1], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 3, 1, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [2, 4, 1], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 2, 4, 1, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 0, 3], [4, 3, 1], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 0, 3, 4, 3, 1, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [5, 1, 3], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 5, 1, 3, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [5, 1, 3], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 5, 1, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 0, 3], [4, 3, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 0, 3, 4, 3, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [5, 1, 3], [4, 0, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 5, 1, 3, 4, 0, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [5, 1, 3], [2, 4, 1], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 5, 1, 3, 2, 4, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [5, 1, 3], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 5, 1, 3, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [4, 0, 3], [2, 4, 1], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 4, 0, 3, 2, 4, 1, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [5, 1, 3], [4, 0, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 5, 1, 3, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[1, 5, 2], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1], [4, 0, 3], [5, 1, 3]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1, 4, 0, 3, 5, 1, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [2, 4, 1], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 2, 4, 1, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [2, 4, 1], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 2, 4, 1, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [4, 3, 1], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 4, 3, 1, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [4, 3, 1], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 4, 3, 1, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [5, 3, 0], [2, 4, 1], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 5, 3, 0, 2, 4, 1, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 2, 0], [5, 3, 0], [4, 3, 1], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 2, 0, 5, 3, 0, 4, 3, 1, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 2, 0], [4, 3, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 2, 0, 4, 3, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 2, 0], [5, 3, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 2, 0, 5, 3, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [4, 3, 1], [5, 3, 0], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 4, 3, 1, 5, 3, 0, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 2, 0], [4, 3, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 2, 0, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [2, 4, 1], [5, 3, 0], [4, 3, 1], [4, 2, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 2, 4, 1, 5, 3, 0, 4, 3, 1, 4, 2, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [4, 2, 0], [2, 4, 1], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 4, 2, 0, 2, 4, 1, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [4, 2, 0], [5, 3, 0], [2, 4, 1]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 4, 2, 0, 5, 3, 0, 2, 4, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[1, 5, 2], [4, 0, 3], [5, 1, 3], [4, 3, 1], [2, 4, 1], [4, 2, 0], [5, 3, 0]}") {
    std::vector<int> T = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 5, 2, 4, 0, 3, 5, 1, 3, 4, 3, 1, 2, 4, 1, 4, 2, 0, 5, 3, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 8", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 3, 8, 6, 7};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 7;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 3, 1], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 3, 1, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [4, 3, 1], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 4, 3, 1, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [4, 3, 1], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 4, 3, 1, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 3, 1], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 3, 1, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 3, 1], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 3, 1, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [4, 3, 1], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 4, 3, 1, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [4, 3, 1], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 4, 3, 1, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 4, 3, 1, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 4, 3, 1, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [0, 4, 7], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 0, 4, 7, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [0, 4, 7], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 0, 4, 7, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [0, 4, 7], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 0, 4, 7, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [0, 4, 7], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 0, 4, 7, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [0, 4, 7], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 0, 4, 7, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [0, 4, 7], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 0, 4, 7, 4, 1, 6, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [0, 4, 7], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 0, 4, 7, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [0, 4, 7], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 0, 4, 7, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [0, 4, 7], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 0, 4, 7, 4, 0, 3, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [0, 4, 7], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 0, 4, 7, 6, 1, 8, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [0, 4, 7], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 0, 4, 7, 1, 3, 8, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 0, 4, 7, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 0, 4, 7, 6, 1, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8, 0, 4, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 8", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 3, 8, 6, 7};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 7;
  short int ulimit = 10;

  SECTION("permutation 1: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[4, 6, 7], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 3, 1], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 3, 1, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[4, 6, 7], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[4, 6, 7], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [4, 3, 1], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 4, 3, 1, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [4, 3, 1], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 4, 3, 1, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[4, 6, 7], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 3, 1], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 3, 1, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 3, 1], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 3, 1, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 3, 1], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 3, 1, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 3, 1], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 3, 1, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [4, 3, 1], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 4, 3, 1, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [4, 3, 1], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 4, 3, 1, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [4, 3, 1], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 4, 3, 1, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [4, 3, 1], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 4, 3, 1, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[4, 6, 7], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[4, 6, 7], [4, 3, 1], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [1, 3, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 1, 3, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [0, 4, 7], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 0, 4, 7, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [1, 3, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 1, 3, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 0, 3], [4, 1, 6], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 0, 3, 4, 1, 6, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [0, 4, 7], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 0, 4, 7, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [1, 3, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 1, 3, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[4, 6, 7], [4, 3, 1], [6, 1, 8], [4, 1, 6], [4, 0, 3], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 6, 1, 8, 4, 1, 6, 4, 0, 3, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 0, 3], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 0, 3, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 1, 6], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 1, 6, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [6, 1, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 6, 1, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [0, 4, 7], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 0, 4, 7, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [0, 4, 7], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 0, 4, 7, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [6, 1, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 6, 1, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [0, 4, 7], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 0, 4, 7, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [0, 4, 7], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 0, 4, 7, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [6, 1, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 6, 1, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[4, 6, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [6, 1, 8], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 6, 1, 8, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [1, 3, 8], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 1, 3, 8, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [4, 1, 6], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 4, 1, 6, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [0, 4, 7], [4, 1, 6], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 0, 4, 7, 4, 1, 6, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [0, 4, 7], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 0, 4, 7, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [1, 3, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 1, 3, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [6, 1, 8], [4, 1, 6], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 6, 1, 8, 4, 1, 6, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [0, 4, 7], [4, 1, 6], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 0, 4, 7, 4, 1, 6, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [6, 1, 8], [4, 1, 6], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 6, 1, 8, 4, 1, 6, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [0, 4, 7], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 0, 4, 7, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [0, 4, 7], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 0, 4, 7, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [6, 1, 8], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 6, 1, 8, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[4, 6, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [6, 1, 8], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 6, 1, 8, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [6, 1, 8], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 6, 1, 8, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [1, 3, 8], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 1, 3, 8, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [1, 3, 8], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 1, 3, 8, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [4, 0, 3], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 4, 0, 3, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [0, 4, 7], [4, 0, 3], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 0, 4, 7, 4, 0, 3, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [1, 3, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 1, 3, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [6, 1, 8], [4, 0, 3], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 6, 1, 8, 4, 0, 3, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [0, 4, 7], [6, 1, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 0, 4, 7, 6, 1, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [0, 4, 7], [4, 0, 3], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 0, 4, 7, 4, 0, 3, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [0, 4, 7], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 0, 4, 7, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [6, 1, 8], [4, 0, 3], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 6, 1, 8, 4, 0, 3, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [0, 4, 7], [6, 1, 8], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 0, 4, 7, 6, 1, 8, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [0, 4, 7], [1, 3, 8], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 0, 4, 7, 1, 3, 8, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [0, 4, 7], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 0, 4, 7, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [6, 1, 8], [1, 3, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 6, 1, 8, 1, 3, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [0, 4, 7], [6, 1, 8]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 0, 4, 7, 6, 1, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[4, 6, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8], [6, 1, 8], [0, 4, 7]}") {
    std::vector<int> T = { 4, 6, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8, 6, 1, 8, 0, 4, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [1, 3, 8], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 1, 3, 8, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 0, 3], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 0, 3, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 0, 3], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 0, 3, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 1, 6], [1, 3, 8], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 1, 6, 1, 3, 8, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 3, 1], [4, 1, 6], [4, 0, 3], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 3, 1, 4, 1, 6, 4, 0, 3, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 0, 3], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 0, 3, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 3, 1], [4, 1, 6], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 3, 1, 4, 1, 6, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 0, 3], [4, 1, 6], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 0, 3, 4, 1, 6, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 3, 1], [4, 0, 3]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 3, 1, 4, 0, 3 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [1, 3, 8], [4, 1, 6], [4, 0, 3], [4, 3, 1]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 1, 3, 8, 4, 1, 6, 4, 0, 3, 4, 3, 1 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 3, 1], [1, 3, 8], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 3, 1, 1, 3, 8, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [4, 3, 1], [4, 1, 6], [1, 3, 8]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 4, 3, 1, 4, 1, 6, 1, 3, 8 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[4, 6, 7], [6, 1, 8], [0, 4, 7], [4, 0, 3], [1, 3, 8], [4, 3, 1], [4, 1, 6]}") {
    std::vector<int> T = { 4, 6, 7, 6, 1, 8, 0, 4, 7, 4, 0, 3, 1, 3, 8, 4, 3, 1, 4, 1, 6 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[CPU] boundary tests: case 9", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 4, 6};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 8;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute(dR.data(), dR_offs, dR_size, head, 
                      T.data(), T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
TEST_CASE("[GPU] boundary tests: case 9", "[boundary]") {

  const short int dR_size = 16;
  std::vector<int> dR(16, 0xff);
  std::vector<int> ans = {0, 2, 4, 6};

  short int head = -1;
  short int dR_offs = 0;
  short int T_offs = 0;
  short int T_size = 8;
  short int ulimit = 10;

  SECTION("permutation 1: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 2: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 3: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 4: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 5: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 6: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 7: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 8: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 9: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 10: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 11: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 12: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 13: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 14: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 15: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 16: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 17: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 18: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 19: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 20: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 21: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 22: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 23: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 24: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 25: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 26: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 27: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 28: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 29: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 30: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 31: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 32: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 33: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 34: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 35: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 36: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 37: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 38: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 39: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 40: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 41: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 42: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 43: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 44: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 45: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 46: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 47: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 48: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 49: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 50: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 51: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 52: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 53: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 54: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 55: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 56: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 57: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 58: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 59: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 60: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 61: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 62: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 63: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 64: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 65: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 66: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 67: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 68: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 69: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 70: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 71: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 72: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 73: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 74: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 75: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 76: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 77: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 78: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 79: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 80: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 81: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 82: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 83: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 84: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 85: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 86: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 87: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 88: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 89: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 90: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 91: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 92: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 93: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 94: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 95: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 96: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 97: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 98: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 99: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 100: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 101: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 102: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 103: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 104: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 105: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 106: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 107: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 108: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 109: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 110: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [1, 6, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 1, 6, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 111: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 112: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 113: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 114: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 115: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 116: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 117: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [1, 6, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 1, 6, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 118: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 119: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [1, 6, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 1, 6, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 120: {[2, 4, 6], [1, 2, 6], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 121: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 122: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 123: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 124: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 125: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 126: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 127: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 128: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 129: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 130: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 131: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 132: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 133: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 134: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 135: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 136: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 137: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 138: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 139: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 140: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 141: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 142: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 143: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 144: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 145: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 146: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 147: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 148: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 149: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 150: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 151: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 152: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 153: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 154: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 155: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 156: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 157: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 158: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 159: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 160: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 161: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 162: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 163: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 164: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 165: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 166: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 167: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 168: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 169: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 170: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 171: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 172: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 173: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 174: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 175: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 176: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 177: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 178: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 179: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 180: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 181: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 182: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 183: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 184: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 185: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 186: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 187: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 188: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 189: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 190: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 191: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 192: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 193: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 194: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 195: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 196: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 197: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 198: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 199: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 200: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 201: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 202: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 203: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 204: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [6, 0, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 6, 0, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 205: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 206: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7], [2, 5, 0], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7, 2, 5, 0, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 207: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 208: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 5, 0], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 5, 0, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 209: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 210: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 211: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 212: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 213: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 214: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 215: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 216: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 217: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 218: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [6, 0, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 6, 0, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 219: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 220: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [0, 5, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 0, 5, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 221: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 222: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [2, 1, 7], [5, 2, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 2, 1, 7, 5, 2, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 223: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 224: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 225: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 226: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [0, 5, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 0, 5, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 227: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 228: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [6, 0, 7], [5, 2, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 6, 0, 7, 5, 2, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 229: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [6, 0, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 6, 0, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 230: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [2, 1, 7], [5, 2, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 2, 1, 7, 5, 2, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 231: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [2, 1, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 2, 1, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 232: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [6, 0, 7], [5, 2, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 6, 0, 7, 5, 2, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 233: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 234: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 235: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [6, 0, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 6, 0, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 236: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [2, 1, 7], [0, 5, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 2, 1, 7, 0, 5, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 237: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [2, 1, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 2, 1, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 238: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [6, 0, 7], [0, 5, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 6, 0, 7, 0, 5, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 239: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [2, 1, 7], [6, 0, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 2, 1, 7, 6, 0, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 240: {[2, 4, 6], [1, 2, 6], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7], [6, 0, 7], [2, 1, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7, 6, 0, 7, 2, 1, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 241: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 242: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 243: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [5, 2, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 5, 2, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 244: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 245: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 246: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 247: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [1, 6, 7], [5, 2, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 1, 6, 7, 5, 2, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 248: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 249: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 250: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [5, 2, 7], [2, 5, 0], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 5, 2, 7, 2, 5, 0, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 251: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [1, 6, 7], [5, 2, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 1, 6, 7, 5, 2, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 252: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [0, 5, 7], [2, 5, 0], [5, 2, 7], [1, 6, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 0, 5, 7, 2, 5, 0, 5, 2, 7, 1, 6, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 253: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [1, 6, 7], [0, 5, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 1, 6, 7, 0, 5, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 254: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [1, 6, 7], [2, 5, 0], [0, 5, 7]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 1, 6, 7, 2, 5, 0, 0, 5, 7 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

  SECTION("permutation 255: {[2, 4, 6], [1, 2, 6], [6, 0, 7], [2, 1, 7], [5, 2, 7], [0, 5, 7], [1, 6, 7], [2, 5, 0]}") {
    std::vector<int> T = { 2, 4, 6, 1, 2, 6, 6, 0, 7, 2, 1, 7, 5, 2, 7, 0, 5, 7, 1, 6, 7, 2, 5, 0 };
    boundary::compute_gpu(dR, dR_offs, dR_size, head,
                          T, T_offs, T_size);
    std::vector<int> vec(dR.begin() + dR_offs, dR.end());
    boundary::test(vec, head, ans, ulimit);
  }

}
