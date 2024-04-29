#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <libsycl.hpp>
#include <planes.hpp>

TEST_CASE("dot product", "[planes]") {
  REQUIRE(planes::dot(1, 2, 3, 4, 5, 6, 7, 8) == 70);
  REQUIRE_THAT(planes::dot(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f), 
               Catch::Matchers::WithinRel(70.0f, 0.001f));
  REQUIRE_THAT(planes::dot(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f), 
               Catch::Matchers::WithinRel(0.0f));
  REQUIRE_THAT(planes::dot(-1.0f,-2.0f,-3.0f,-4.0f,-5.0f,-6.0f,-7.0f,-8.0f),
               Catch::Matchers::WithinRel(70.0f, 0.001f));
  REQUIRE_THAT(planes::dot(-1.0f, -2.0f, -3.0f, -4.0f, 5.0f, 6.0f, 7.0f, 8.0f), 
               Catch::Matchers::WithinRel(-70.0f, 0.001f));
}

TEST_CASE("intersect", "[planes]") {
  const double rel = 0.01;
  double p0, p1, p2, p3;
  SECTION("Case 1") {
    planes::intersect<double>(p0, p1, p2, p3, -7.26915845, -6.01161514,
      7.13903569, 4.25780487, 2.18863785, 8.46638014, 9.08512613, 0.54271548,
      -2.40110545, 5.68121716, 7.12594416, 1.48054489);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(0.2224792, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(0.14461607, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.24809964, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 2") {
    planes::intersect<double>(p0, p1, p2, p3, -6.45188017, 8.06335492,
      -6.5427573, -2.99987203, 5.26738272, -7.64728783, 1.55136299, 9.0587996,
      0.49473855, -9.46365599, -8.00974195, -6.96137158);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-7.47124504, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-3.41334152, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(2.70233493, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 3") {
    planes::intersect<double>(p0, p1, p2, p3, -0.98188758, 3.96775056,
      9.35059983, 0.60976752, -4.23027845, 7.22467625, -4.20916011,
      -2.99469027, -4.05069284, 7.07849567, 9.23348395, 8.46712423);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(6.11440371, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(3.47233684, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.89657103, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 4") {
    planes::intersect<double>(p0, p1, p2, p3, -8.10038267, -5.68548483,
      -3.17091839, 9.54064891, 3.23329045, -3.19175926, 6.66274889,
      -6.35849152, -2.01723377, -2.68522449, -5.63216812, 7.62769942);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(0.46539981, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(0.48036138, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(0.95860084, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 5") {
    planes::intersect<double>(p0, p1, p2, p3, -8.67532066, 2.58425607,
      7.27430838, 7.04316317, 4.06235503, 1.22814697, 9.83399229, 6.74962755,
      0.06390301, 6.10761669, -5.83292943, -5.51293143);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(0.19699621, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(0.14953031, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.78640913, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 6") {
    planes::intersect<double>(p0, p1, p2, p3, 8.65384588, 7.65276696,
      8.83076072, 2.16007184, 4.5371244, 1.39808981, 8.48101964, 9.03284945,
      -5.28078884, -4.11232108, -8.26587303, 6.16801865);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-13.28673789, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(9.59446145, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(4.46135207, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 7") {
    planes::intersect<double>(p0, p1, p2, p3, 5.72158955, -1.83804275,
      -3.99566531, -0.7260049, -5.00300389, -2.35738558, 8.60082608,
      -7.59861043, -8.02983076, -6.85415036, -7.68776988, 8.81260021);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(0.65658917, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-0.69050893, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(1.07614463, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 8") {
    planes::intersect<double>(p0, p1, p2, p3, -5.08885204, 6.81858135,
      -9.62889368, -8.91963827, -9.62136386, 5.59220333, 4.54587427,
      9.94841461, 9.3934439, -0.8974448, 7.44166019, 1.94100189);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(0.68255252, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(0.28026757, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-1.08860064, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 8") {
    planes::intersect<double>(p0, p1, p2, p3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      0);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 9") {
    planes::intersect<double>(p0, p1, p2, p3, 1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1,
      -1);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-1.0, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-2.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(1.0, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 11") {
    planes::intersect<double>(p0, p1, p2, p3, 1, 1, 0, 1, 0, 1, 1, 2, 1, 0, 1,
      3);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-1.0, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-2.0, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 12") {
    planes::intersect<double>(p0, p1, p2, p3, -1, 0, 0, -1, 0, -1, 0, -2, 0, 0,
      -1, 1);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-1.0, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-2.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(1.0, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 13") {
    planes::intersect<double>(p0, p1, p2, p3, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0.001,
      1, 0.001);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.001, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

  SECTION("Case 14") {
    planes::intersect<double>(p0, p1, p2, p3, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1,
      0.5, 1, 1, 1);
    REQUIRE_THAT(p0, Catch::Matchers::WithinRel(-0.66666667, rel));
    REQUIRE_THAT(p1, Catch::Matchers::WithinRel(-0.0, rel));
    REQUIRE_THAT(p2, Catch::Matchers::WithinRel(-0.66666667, rel));
    REQUIRE_THAT(p3, Catch::Matchers::WithinRel(1, rel));
  }

}

