#include <catch2/catch_test_macros.hpp>
#include <status.hpp>

TEST_CASE("State status manipulation", "[status]") {
  cc::state state;

  SECTION("Initial state") {
    for (int s = 0; s < cc::status_enum_size; ++s) {
      REQUIRE_FALSE(state.get(static_cast<cc::status>(s)));
    }
  }

  SECTION("Setting individual statuses to true") {
    for (int s = 0; s < cc::status_enum_size; ++s) {
      state.set_true(static_cast<cc::status>(s));
      REQUIRE(state.get(static_cast<cc::status>(s)));
      state.reset();
    }
  }

  SECTION("Setting individual statuses to false after setting them to true") {
    for (int s = 0; s < cc::status_enum_size; ++s) {
      state.set_true(static_cast<cc::status>(s));
      state.set_false(static_cast<cc::status>(s));
      REQUIRE_FALSE(state.get(static_cast<cc::status>(s)));
      state.reset();
    }
  }

  SECTION("Independent status manipulation") {
    for (int s = 0; s < cc::status_enum_size; ++s) {
      state.set_true(static_cast<cc::status>(s));
      REQUIRE(state.get(static_cast<cc::status>(s)));

      for (int other = 0; other < cc::status_enum_size; ++other) {
        if (other != s) {
          REQUIRE_FALSE(state.get(static_cast<cc::status>(other)));
        }
      }

      state.reset();
    }
  }
}
