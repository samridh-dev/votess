#include <catch2/catch_test_macros.hpp> 
#include <catch2/matchers/catch_matchers_floating_point.hpp> 
#include <heap.hpp>
#include <vector>
#include <utility>
#include <limits>

///////////////////////////////////////////////////////////////////////////////
/// heap:swap()                                                             ///
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("[GPU] [float] heap::swap", "[heap]") {

  sycl::queue q;

  std::vector<int> hid = { 1, 4, 7, 
                           2, 5, 8,
                           3, 6, 9 };

  std::vector<float> hpq = { 1.0f, 4.0f, 7.0f, 
                             2.0f, 5.0f, 8.0f,
                             3.0f, 6.0f, 9.0f };

  const int rowsize = 3;

  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
  sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));

  SECTION("case: 1") {

    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 1);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 2);
    REQUIRE(rhid[rowsize * 1 + index] == 1);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(2.0f));
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(1.0f));

  }

  SECTION("case: 2") {

    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 1, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 1 + index] == 6);
    REQUIRE(rhid[rowsize * 2 + index] == 5);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(6.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(5.0f));

  }

  SECTION("case: 3") {

    int index = 2;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 9);
    REQUIRE(rhid[rowsize * 2 + index] == 7);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(9.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(7.0f));

  }

  SECTION("case: 4") {

    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 0);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 1);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(1.0f));
    REQUIRE(rhid[rowsize * 1 + index] == 2);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(2.0f));
    REQUIRE(rhid[rowsize * 2 + index] == 3);
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(3.0f));

  }

  SECTION("case: 5") {

    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_5>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 6);
    REQUIRE(rhid[rowsize * 2 + index] == 4);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(6.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(4.0f));

  }

  SECTION("case: 6") {

    int index = 2;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_swap_6>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 1, 1);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 1 + index] == 8);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(8.0f));

  }

}

TEST_CASE("[GPU] [double] heap::swap", "[heap]") {

  sycl::queue q;

  if (!q.get_device().has(sycl::aspect::fp64)) {
    REQUIRE(1==1);
    return; // Skip the test if fp64 is not supported
  }

  std::vector<int> hid = { 1, 4, 7, 
                           2, 5, 8,
                           3, 6, 9 };

  std::vector<double> hpq = { 1.0f, 4.0f, 7.0f, 
                             2.0f, 5.0f, 8.0f,
                             3.0f, 6.0f, 9.0f };

  const int rowsize = 3;

  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
  sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));

  SECTION("case: 1") {

    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 1);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 2);
    REQUIRE(rhid[rowsize * 1 + index] == 1);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(2.0f));
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(1.0f));

  }

  SECTION("case: 2") {

    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 1, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 1 + index] == 6);
    REQUIRE(rhid[rowsize * 2 + index] == 5);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(6.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(5.0f));

  }

  SECTION("case: 3") {

    int index = 2;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 9);
    REQUIRE(rhid[rowsize * 2 + index] == 7);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(9.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(7.0f));

  }

  SECTION("case: 4") {

    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 0);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 1);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(1.0f));
    REQUIRE(rhid[rowsize * 1 + index] == 2);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(2.0f));
    REQUIRE(rhid[rowsize * 2 + index] == 3);
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(3.0f));

  }

  SECTION("case: 5") {

    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_5>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 0, 2);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 0 + index] == 6);
    REQUIRE(rhid[rowsize * 2 + index] == 4);
    REQUIRE_THAT(rhpq[rowsize * 0 + index], Catch::Matchers::WithinRel(6.0f));
    REQUIRE_THAT(rhpq[rowsize * 2 + index], Catch::Matchers::WithinRel(4.0f));

  }

  SECTION("case: 6") {

    int index = 2;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_swap_6>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::swap(did, dpq, rowsize, index, 1, 1);
      });
      q.wait();
    });

    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();

    REQUIRE(rhid[rowsize * 1 + index] == 8);
    REQUIRE_THAT(rhpq[rowsize * 1 + index], Catch::Matchers::WithinRel(8.0f));

  }

}

TEST_CASE("[CPU] [float] heap::swap", "[heap]") {

  std::vector<int> hid = {1, 2};
  std::vector<float> hpq = {3.5f, 4.1f};

  SECTION("case: 1") {
    heap::swap(hid, hpq, 0, 0, 1);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("case: 2") {
    heap::swap(hid, hpq, 0, 1, 0);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("case: 3") {
    heap::swap(hid, hpq, 0, 1, 1);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }

  SECTION("case: 4") {
    heap::swap(hid, hpq, 0, 0, 0);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }

}

TEST_CASE("[CPU] [double] heap::swap", "[heap]") {

  std::vector<int> hid = {1, 2};
  std::vector<double> hpq = {3.5f, 4.1f};

  SECTION("case: 1") {
    heap::swap(hid, hpq, 0, 0, 1);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("case: 2") {
    heap::swap(hid, hpq, 0, 1, 0);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("case: 3") {
    heap::swap(hid, hpq, 0, 1, 1);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }

  SECTION("case: 4") {
    heap::swap(hid, hpq, 0, 0, 0);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }

}

///////////////////////////////////////////////////////////////////////////////
/// heap::maxheapify()                                                      ///
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("[GPU] [float] heap::maxheapify", "[heap]") {

  sycl::queue q;

  std::vector<int> hid = { 01, 07, 13,
                           02, 07, 14,
                           03, 07, 15,
                           04, 10, 16,  
                           05, 11, 17, 
                           06, 12, 18 }; 

  const int rowsize = 3;
  const int colsize = 6;

  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));

  SECTION("case: 1") {

    std::vector<float> hpq = { 05.0f, 09.0f, 13.0f,
                               04.0f, 08.0f, 14.0f,
                               03.0f, 07.0f, 15.0f,
                               02.0f, 10.0f, 16.0f,  
                               01.0f, 11.0f, 17.0f, 
                               06.0f, 12.0f, 18.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));

    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 2") {

    std::vector<float> hpq = { 18.0f, 17.0f, 16.0f,
                               15.0f, 14.0f, 13.0f,
                               12.0f, 11.0f, 10.0f,
                               09.0f, 08.0f, 07.0f,  
                               06.0f, 05.0f, 04.0f, 
                               03.0f, 02.0f, 01.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;  // Second column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 3") {

    std::vector<float> hpq = { 10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f, 
                               10.0f, 10.0f, 10.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 2;  // Third column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] == 10.0f);
    REQUIRE(rhpq[rowsize * 1 + index] == 10.0f);
    REQUIRE(rhpq[rowsize * 2 + index] == 10.0f);
  }

  SECTION("case: 4") {

    std::vector<float> hpq = { 01.0f, 02.0f, 03.0f,
                               10.0f, 20.0f, 30.0f,
                               05.0f, 15.0f, 25.0f,
                               07.0f, 17.0f, 27.0f,  
                               06.0f, 16.0f, 26.0f, 
                               09.0f, 19.0f, 29.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 5") {

    std::vector<float> hpq = { 10.0f, 20.0f, 30.0f,
                               05.0f, 15.0f, 25.0f,
                               07.0f, 17.0f, 27.0f,  
                               06.0f, 16.0f, 26.0f, 
                               09.0f, 19.0f, 29.0f,
                               04.0f, 14.0f, 24.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 2;  // Last column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_5>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 6") {

    std::vector<float> hpq = { 30.0f, 29.0f, 28.0f,
                               27.0f, 26.0f, 25.0f,
                               24.0f, 23.0f, 22.0f,
                               21.0f, 20.0f, 19.0f,
                               18.0f, 17.0f, 16.0f, 
                               15.0f, 14.0f, 13.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_6>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 7 - heapify with epsilon values") {

    float epsilon = std::numeric_limits<float>::epsilon();

    std::vector<float> hpq = { 3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class fcase_maxheapify_7>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);

  }

}

TEST_CASE("[GPU] [double] heap::maxheapify", "[heap]") {

  sycl::queue q;
  if (!q.get_device().has(sycl::aspect::fp64)) {
    REQUIRE(1==1);
    return; // Skip the test if fp64 is not supported
  }

  std::vector<int> hid = { 01, 07, 13,
                           02, 07, 14,
                           03, 07, 15,
                           04, 10, 16,  
                           05, 11, 17, 
                           06, 12, 18 }; 

  const int rowsize = 3;
  const int colsize = 6;

  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));

  SECTION("case: 1") {

    std::vector<double> hpq = { 05.0f, 09.0f, 13.0f,
                               04.0f, 08.0f, 14.0f,
                               03.0f, 07.0f, 15.0f,
                               02.0f, 10.0f, 16.0f,  
                               01.0f, 11.0f, 17.0f, 
                               06.0f, 12.0f, 18.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));

    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 2") {

    std::vector<double> hpq = { 18.0f, 17.0f, 16.0f,
                               15.0f, 14.0f, 13.0f,
                               12.0f, 11.0f, 10.0f,
                               09.0f, 08.0f, 07.0f,  
                               06.0f, 05.0f, 04.0f, 
                               03.0f, 02.0f, 01.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;  // Second column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 3") {

    std::vector<double> hpq = { 10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f,
                               10.0f, 10.0f, 10.0f, 
                               10.0f, 10.0f, 10.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 2;  // Third column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] == 10.0f);
    REQUIRE(rhpq[rowsize * 1 + index] == 10.0f);
    REQUIRE(rhpq[rowsize * 2 + index] == 10.0f);
  }

  SECTION("case: 4") {

    std::vector<double> hpq = { 01.0f, 02.0f, 03.0f,
                               10.0f, 20.0f, 30.0f,
                               05.0f, 15.0f, 25.0f,
                               07.0f, 17.0f, 27.0f,  
                               06.0f, 16.0f, 26.0f, 
                               09.0f, 19.0f, 29.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 5") {

    std::vector<double> hpq = { 10.0f, 20.0f, 30.0f,
                               05.0f, 15.0f, 25.0f,
                               07.0f, 17.0f, 27.0f,  
                               06.0f, 16.0f, 26.0f, 
                               09.0f, 19.0f, 29.0f,
                               04.0f, 14.0f, 24.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 2;  // Last column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_5>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 6") {

    std::vector<double> hpq = { 30.0f, 29.0f, 28.0f,
                               27.0f, 26.0f, 25.0f,
                               24.0f, 23.0f, 22.0f,
                               21.0f, 20.0f, 19.0f,
                               18.0f, 17.0f, 16.0f, 
                               15.0f, 14.0f, 13.0f };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;  // First column

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_6>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);
  }

  SECTION("case: 7 - heapify with epsilon values") {

    double epsilon = std::numeric_limits<double>::epsilon();

    std::vector<double> hpq = { 3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon,
                               3 * epsilon, 2 * epsilon, epsilon };

    sycl::buffer<double> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class dcase_maxheapify_7>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<double> dpq(ahpq);
        heap::maxheapify(did, dpq, rowsize, index, colsize, 0);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 1 + index]);
    REQUIRE(rhpq[rowsize * 0 + index] >= rhpq[rowsize * 2 + index]);

  }

}

TEST_CASE("[CPU] [float] heap::maxheapify", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {0.0f, 32.0f, 0.01f, 32.0f};

    SECTION("case: 1") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
      REQUIRE(hpq[0] == 32.0f);
      REQUIRE(hpq[1] <= hpq[0]);
      REQUIRE(hpq[2] <= hpq[0]);
    }

    SECTION("case: 2") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 1);
      REQUIRE(hpq[1] == 32.0f);
      REQUIRE(hpq[3] <= hpq[1]);
    }
  }

  SECTION("single element") {
    std::vector<int> hid = {4};
    std::vector<float> hpq = {0.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid.size() == 1);
    REQUIRE(hpq[0] == 0.0f);
  }

  SECTION("equal elements") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<float> hpq = {32.0f, 32.0f, 32.0f, 32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid[0] == 1);
    REQUIRE(hpq[0] == 32.0f);
  }

  SECTION("correct order") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {32.0f, 0.01f, 0.0f, -32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hpq[0] >= hpq[1]);
    REQUIRE(hpq[0] >= hpq[2]);
  }

}

TEST_CASE("[CPU] [double] heap::maxheapify", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<double> hpq = {0.0f, 32.0f, 0.01f, 32.0f};

    SECTION("case: 1") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
      REQUIRE(hpq[0] == 32.0f);
      REQUIRE(hpq[1] <= hpq[0]);
      REQUIRE(hpq[2] <= hpq[0]);
    }

    SECTION("case: 2") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 1);
      REQUIRE(hpq[1] == 32.0f);
      REQUIRE(hpq[3] <= hpq[1]);
    }
  }

  SECTION("single element") {
    std::vector<int> hid = {4};
    std::vector<double> hpq = {0.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid.size() == 1);
    REQUIRE(hpq[0] == 0.0f);
  }

  SECTION("equal elements") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<double> hpq = {32.0f, 32.0f, 32.0f, 32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid[0] == 1);
    REQUIRE(hpq[0] == 32.0f);
  }

  SECTION("correct order") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<double> hpq = {32.0f, 0.01f, 0.0f, -32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hpq[0] >= hpq[1]);
    REQUIRE(hpq[0] >= hpq[2]);
  }

}

///////////////////////////////////////////////////////////////////////////////
/// heap::sort()                                                            ///
///////////////////////////////////////////////////////////////////////////////

TEST_CASE("[GPU] heap::sort", "[heap]") {

  sycl::queue q;

  std::vector<int> hid = { 01, 07, 13,
                           02, 07, 14,
                           03, 07, 15,
                           04, 10, 16,  
                           05, 11, 17, 
                           06, 12, 18 }; 

  const int rowsize = 3;
  const int colsize = 6;

  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));

  SECTION("case: 1") {

    std::vector<float> hpq = { 5.0f, 9.0f, 13.0f,
                               4.0f, 8.0f, 14.0f,
                               3.0f, 7.0f, 15.0f,
                               2.0f, 10.0f, 16.0f,  
                               1.0f, 11.0f, 17.0f, 
                               6.0f, 12.0f, 18.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

  SECTION("case: 2") {

    std::vector<float> hpq = { 1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f,
                               7.0f, 8.0f, 9.0f,
                               10.0f, 11.0f, 12.0f,  
                               13.0f, 14.0f, 15.0f, 
                               16.0f, 17.0f, 18.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

  SECTION("case: 3") {

    std::vector<float> hpq = { 18.0f, 17.0f, 16.0f,
                               15.0f, 14.0f, 13.0f,
                               12.0f, 11.0f, 10.0f,
                               9.0f, 8.0f, 7.0f,  
                               6.0f, 5.0f, 4.0f, 
                               3.0f, 2.0f, 1.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 2;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

  SECTION("case: 4") {

    std::vector<float> hpq = { 10.0f, 10.0f, 10.0f,
                               8.0f, 8.0f, 8.0f,
                               6.0f, 6.0f, 6.0f,
                               4.0f, 4.0f, 4.0f,  
                               2.0f, 2.0f, 2.0f, 
                               1.0f, 1.0f, 1.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

  SECTION("case: 5") {

    std::vector<float> hpq = { 18.0f, 1.0f, 13.0f,
                               18.0f, 1.0f, 14.0f,
                               18.0f, 1.0f, 15.0f,
                               18.0f, 1.0f, 16.0f,  
                               18.0f, 1.0f, 17.0f, 
                               18.0f, 1.0f, 18.0f };

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 1;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_5>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

  SECTION("case: 6") {

    std::vector<float> hpq = { 5.0f, 9.0f, 13.0f,
                               4.0f, 8.0f, 14.0f,
                               3.0f, 7.0f, 15.0f,
                               2.0f, 10.0f, 16.0f,  
                               1.0f, 11.0f, 17.0f, 
                               6.0f, 12.0f, 18.0f };
   
    for (auto& it : hpq) it *= std::numeric_limits<float>::epsilon();

    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    int index = 0;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class heapsort_6>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::sort(did, dpq, rowsize, index, colsize);
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();

    for (int i = 0; i < colsize - 1; ++i) {
      REQUIRE(rhpq[rowsize * i + index] <= rhpq[rowsize * (i + 1) + index]);
    }
  }

}

TEST_CASE("[CPU] heap::sort", "[heap]") {

  SECTION("regular") {

    std::vector<int> hid = {1, 4, 2, 3};
    std::vector<int> hpq = {4, 1, 3, 2};

    heap::sort(hid, hpq, 0, hid.size());

    REQUIRE(hpq[0] <= hpq[1]);
    REQUIRE(hpq[1] <= hpq[2]);
    REQUIRE(hpq[2] <= hpq[3]);

  }

  SECTION("single element") {

    std::vector<int> hid = {4};
    std::vector<int> hpq = {32};

    heap::sort(hid, hpq, 0, hid.size());

    REQUIRE(hpq.size() == 1);
    REQUIRE(hpq[0] == 32);

  }

  SECTION("equal elements") {

    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<int> hpq = {32, 32, 32, 32};

    heap::sort(hid, hpq, 0, hid.size());

    for (size_t i = 0; i < hpq.size() - 1; ++i) {
      REQUIRE(hpq[i] == hpq[i + 1]);
    }

  }

  SECTION("sorted elements") {

    std::vector<int> hid = {1, 2, 3, 4};
    std::vector<int> hpq = {1, 2, 3, 4};

    heap::sort(hid, hpq, 0, hid.size());

    for (size_t i = 0; i < hpq.size() - 1; ++i) {
      REQUIRE(hpq[i] <= hpq[i + 1]);
    }

  }

}

///////////////////////////////////////////////////////////////////////////////
