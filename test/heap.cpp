#include <catch2/catch_test_macros.hpp> 
#include <catch2/matchers/catch_matchers_floating_point.hpp> 
#include <libsycl.hpp>

#include <heap.hpp>

#include <vector>
#include <utility>

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// heap:swap()                                                             ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("heap::swap", "[heap]") {
  std::vector<int> hid = {1,2};
  std::vector<float> hpq = {3.5f,4.1f};
  sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
  sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
  sycl::queue q;
  
  SECTION("test 1") {
    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testswap_1>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, 0, 0, 1);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid[0] == 2);
    REQUIRE_THAT(rhpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(rhid[1] == 1);
    REQUIRE_THAT(rhpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("test 2") {
    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testswap_2>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, 0, 1, 0);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid[0] == 2);
    REQUIRE_THAT(rhpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(rhid[1] == 1);
    REQUIRE_THAT(rhpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("test 3") {
    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testswap_3>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, 0, 1, 1);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid[0] == 1);
    REQUIRE_THAT(rhpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(rhid[1] == 2);
    REQUIRE_THAT(rhpq[1], Catch::Matchers::WithinRel(4.1f));
  }

  SECTION("test 4") {
    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testswap_4>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::swap(did, dpq, 0, 0, 0);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid[0] == 1);
    REQUIRE_THAT(rhpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(rhid[1] == 2);
    REQUIRE_THAT(rhpq[1], Catch::Matchers::WithinRel(4.1f));
  }

}

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("heap::swap - CPU", "[heap]") {
  std::vector<int> hid = {1, 2};
  std::vector<float> hpq = {3.5f, 4.1f};

  SECTION("test 1") {
    heap::swap(hid, hpq, 0, 0, 1);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("test 2") {
    heap::swap(hid, hpq, 0, 1, 0);
    REQUIRE(hid[0] == 2);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(4.1f));
    REQUIRE(hid[1] == 1);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(3.5f));
  }

  SECTION("test 3") {
    heap::swap(hid, hpq, 0, 1, 1);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }

  SECTION("test 4") {
    heap::swap(hid, hpq, 0, 0, 0);
    REQUIRE(hid[0] == 1);
    REQUIRE_THAT(hpq[0], Catch::Matchers::WithinRel(3.5f));
    REQUIRE(hid[1] == 2);
    REQUIRE_THAT(hpq[1], Catch::Matchers::WithinRel(4.1f));
  }
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// heap::maxheapify()                                                      ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("heap::maxheapify", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {0.0f, 32.0f, 0.01f, 32.0f};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    SECTION("Test 1") {
      q.submit([&](sycl::handler& h) {
        auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
        auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
        h.single_task<class testmaxheapify_1>([=]() {
          device_accessor_readwrite_t<int> did(ahid);
          device_accessor_readwrite_t<float> dpq(ahpq);
          heap::maxheapify(did, dpq, 0, 0, 0);
        });
        q.wait();
      });
      auto rhpq = bhpq.get_host_access();
      REQUIRE(rhpq[0] < rhpq[1]);
      REQUIRE(rhpq[0] < rhpq[2]);
    }

    SECTION("Test 2") {
      q.submit([&](sycl::handler& h) {
        auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
        auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
        h.single_task<class testmaxheapify_2>([=]() {
          device_accessor_readwrite_t<int> did(ahid);
          device_accessor_readwrite_t<float> dpq(ahpq);
          heap::maxheapify(did, dpq, 0, did.size(), 0);
        });
        q.wait();
      });
      auto rhpq = bhpq.get_host_access();
      REQUIRE(rhpq[0] >= rhpq[1]);
      REQUIRE(rhpq[0] >= rhpq[2]);
    }
  }

  SECTION("Heap with single element") {
    std::vector<int> hid = {4};
    std::vector<float> hpq = {0.0f};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testmaxheapify_single_element>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, 0, 0, 0);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid.size() == 1);
    REQUIRE(rhpq[0] == 0.0f);
  }

  SECTION("Heap with all elements the same") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<float> hpq = {32.0f, 32.0f, 32.0f, 32.0f};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testmaxheapify_all_elements_same>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, 0, did.size(), 0);
      });
      q.wait();
    });
    auto rhid = bhid.get_host_access();
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhid[0] == 1);
    REQUIRE(rhpq[0] == 32.0f);
  }

  SECTION("Max Heap already in correct order") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {32.0f, 0.01f, 0.0f, -32.0f};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testmaxheapify_correct_order>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, 0, did.size(), 0);
      });
      q.wait();
    });
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[0] >= rhpq[1]);
    REQUIRE(rhpq[0] >= rhpq[2]);
  }

#if false
  SECTION("Calling maxheapify with root index out of bounds") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {32.0f, 0.01f, 0.0f, -32.0f};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<float> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testmaxheapify_out_of_bounds>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<float> dpq(ahpq);
        heap::maxheapify(did, dpq, 0, did.size(), hid.size());
      });
      q.wait();
    });
    // auto rhid = bhid.get_host_access();
    // auto rhpq = bhpq.get_host_access();
    // Normally, you should handle this case within your maxheapify function to
    // prevent out-of-bounds access. This test assumes the function safely
    // handles out-of-bounds indices. No REQUIRE statement here since behavior
    // might depend on your function's implementation details.
  }
#endif

}

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */
TEST_CASE("heap::maxheapify - CPU", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {0.0f, 32.0f, 0.01f, 32.0f};

    SECTION("Test 1") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
      REQUIRE(hpq[0] == 32.0f);
      REQUIRE(hpq[1] <= hpq[0]);
      REQUIRE(hpq[2] <= hpq[0]);
    }

    SECTION("Test 2") {
      heap::maxheapify(hid, hpq, 0, hpq.size(), 1);
      REQUIRE(hpq[1] == 32.0f);
      REQUIRE(hpq[3] <= hpq[1]);
    }
  }

  SECTION("Heap with single element") {
    std::vector<int> hid = {4};
    std::vector<float> hpq = {0.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid.size() == 1);
    REQUIRE(hpq[0] == 0.0f);
  }

  SECTION("Heap with all elements the same") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<float> hpq = {32.0f, 32.0f, 32.0f, 32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hid[0] == 1);
    REQUIRE(hpq[0] == 32.0f);
  }

  SECTION("Max Heap already in correct order") {
    std::vector<int> hid = {4, 3, 2, 1};
    std::vector<float> hpq = {32.0f, 0.01f, 0.0f, -32.0f};

    heap::maxheapify(hid, hpq, 0, hpq.size(), 0);
    REQUIRE(hpq[0] >= hpq[1]);
    REQUIRE(hpq[0] >= hpq[2]);
  }

}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// heap::sort()                                                            ///
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/* Device: [GPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("heap::sort", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {1, 4, 2, 3};
    std::vector<int> hpq = {4, 1, 3, 2};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<int> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testHeapSort>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<int> dpq(ahpq);
        heap::sort(did,dpq,0,did.size());
      });
      q.wait();
    });
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq[0] <= rhpq[1]);
    REQUIRE(rhpq[1] <= rhpq[2]);
    REQUIRE(rhpq[2] <= rhpq[3]);
  }

#if false
  SECTION("Empty Heap") {
    std::vector<int> hid = {};
    std::vector<int> hpq = {};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<int> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testHeapSortEmpty>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<int> dpq(ahpq);
        heap::sort(did, dpq, 0, did.size());
      });
      q.wait();
    });
    // Verifications would depend on expected behavior with empty inputs
  }
#endif

  SECTION("Heap with Single Element") {
    std::vector<int> hid = {4};
    std::vector<int> hpq = {32};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<int> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testHeapSortSingleElement>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<int> dpq(ahpq);
        heap::sort(did, dpq, 0, did.size());
      });
      q.wait();
    });
    auto rhpq = bhpq.get_host_access();
    REQUIRE(rhpq.size() == 1);
    REQUIRE(rhpq[0] == 32);
  }

  SECTION("Heap with All Elements the Same") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<int> hpq = {32, 32, 32, 32};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<int> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testHeapSortAllElementsSame>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<int> dpq(ahpq);
        heap::sort(did, dpq, 0, did.size());
      });
      q.wait();
    });
    auto rhpq = bhpq.get_host_access();
    for (int i = 0; i < static_cast<int>(rhpq.size()) - 1; ++i) {
      REQUIRE(rhpq[i] == rhpq[i + 1]);
    }
  }

// this test is weird
#if false
  SECTION("Already Sorted Heap") {
    std::vector<int> hid = {1, 2, 3, 4};
    std::vector<int> hpq = {1, 2, 3, 4};
    sycl::buffer<int> bhid(hid.data(), sycl::range<1>(hid.size()));
    sycl::buffer<int> bhpq(hpq.data(), sycl::range<1>(hpq.size()));
    sycl::queue q;

    q.submit([&](sycl::handler& h) {
      auto ahid = bhid.get_access<sycl::access::mode::read_write>(h);
      auto ahpq = bhpq.get_access<sycl::access::mode::read_write>(h);
      h.single_task<class testHeapSortAlreadySorted>([=]() {
        device_accessor_readwrite_t<int> did(ahid);
        device_accessor_readwrite_t<int> dpq(ahpq);
        heap::sort(did, dpq, 0, did.size());
      });
      q.wait();
    });

    auto rhpq = bhpq.get_host_access();
    for (int i = 0; i < static_cast<int>(rhpq.size()) - 1; ++i) {
      REQUIRE(rhpq[i] <= rhpq[i + 1]);
    }
  }
#endif

}

/* ------------------------------------------------------------------------- */
/* Device: [CPU]                                                             */
/* ------------------------------------------------------------------------- */

TEST_CASE("heap::sort - CPU", "[heap]") {

  SECTION("regular") {
    std::vector<int> hid = {1, 4, 2, 3};
    std::vector<int> hpq = {4, 1, 3, 2};

    heap::sort(hid, hpq, 0, hid.size());
    REQUIRE(hpq[0] <= hpq[1]);
    REQUIRE(hpq[1] <= hpq[2]);
    REQUIRE(hpq[2] <= hpq[3]);
  }

  SECTION("Heap with Single Element") {
    std::vector<int> hid = {4};
    std::vector<int> hpq = {32};

    heap::sort(hid, hpq, 0, hid.size());
    REQUIRE(hpq.size() == 1);
    REQUIRE(hpq[0] == 32);
  }

  SECTION("Heap with All Elements the Same") {
    std::vector<int> hid = {1, 1, 1, 1};
    std::vector<int> hpq = {32, 32, 32, 32};

    heap::sort(hid, hpq, 0, hid.size());
    for (size_t i = 0; i < hpq.size() - 1; ++i) {
      REQUIRE(hpq[i] == hpq[i + 1]);
    }
  }

  SECTION("Already Sorted Heap") {
    std::vector<int> hid = {1, 2, 3, 4};
    std::vector<int> hpq = {1, 2, 3, 4};

    heap::sort(hid, hpq, 0, hid.size());
    for (size_t i = 0; i < hpq.size() - 1; ++i) {
      REQUIRE(hpq[i] <= hpq[i + 1]);
    }
  }

}

///////////////////////////////////////////////////////////////////////////////
