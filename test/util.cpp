#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

#include <utils.hpp>

TEST_CASE("utils::bmax works with integer values", "[utils::bmax]") {
  REQUIRE(utils::bmax(1, 2) == 2);
  REQUIRE(utils::bmax(2, 1) == 2);
  REQUIRE(utils::bmax(-1, -2) == -1);
  REQUIRE(utils::bmax(-2, -1) == -1);
  REQUIRE(utils::bmax(0, 0) == 0);
}

TEST_CASE("utils::bmin works with integer values", "[utils::bmin]") {
  REQUIRE(utils::bmin(1, 2) == 1);
  REQUIRE(utils::bmin(2, 1) == 1);
  REQUIRE(utils::bmin(-1, -2) == -2);
  REQUIRE(utils::bmin(-2, -1) == -2);
  REQUIRE(utils::bmin(0, 0) == 0);
}

TEST_CASE("utils::bmax works with floating-point values", "[utils::bmax]") {
  REQUIRE_THAT(utils::bmax(1.5, 2.5), Catch::Matchers::WithinRel(2.5));
  REQUIRE_THAT(utils::bmax(-1.5, -2.5), Catch::Matchers::WithinRel(-1.5));
  REQUIRE_THAT(utils::bmax(1.5, 1.5), Catch::Matchers::WithinRel(1.5));
  REQUIRE_THAT(utils::bmax(0.1, 0.2), Catch::Matchers::WithinRel(0.2));
}

TEST_CASE("utils::bmin works with floating-point values", "[utils::bmin]") {
  REQUIRE_THAT(utils::bmin(1.5, 2.5), Catch::Matchers::WithinRel(1.5));
  REQUIRE_THAT(utils::bmin(-1.5, -2.5), Catch::Matchers::WithinRel(-2.5));
  REQUIRE_THAT(utils::bmin(1.5, 1.5), Catch::Matchers::WithinRel(1.5));
  REQUIRE_THAT(utils::bmin(0.1, 0.2), Catch::Matchers::WithinRel(0.1));
}

TEST_CASE("utils::bmax with extreme int values", "[utils::bmax]") {
  REQUIRE(utils::bmax(std::numeric_limits<int>::max(), 0) 
          == std::numeric_limits<int>::max());
  REQUIRE(utils::bmax(0, std::numeric_limits<int>::max()) 
          == std::numeric_limits<int>::max());
  REQUIRE(utils::bmax(std::numeric_limits<int>::min(), 
                      std::numeric_limits<int>::max()) 
          == std::numeric_limits<int>::max());
}

TEST_CASE("utils::bmin with extreme int values", "[utils::bmin]") {
  REQUIRE(utils::bmin(std::numeric_limits<int>::min(), 0) 
          == std::numeric_limits<int>::min());
  REQUIRE(utils::bmin(0, std::numeric_limits<int>::min()) 
          == std::numeric_limits<int>::min());
  REQUIRE(utils::bmin(std::numeric_limits<int>::min(), 
                      std::numeric_limits<int>::max()) 
          == std::numeric_limits<int>::min());
}

TEST_CASE("utils::bmax with extreme float values", "[utils::bmax]") {
  REQUIRE_THAT(utils::bmax(std::numeric_limits<double>::max(), 0.0),
               Catch::Matchers::WithinRel(std::numeric_limits<double>::max()));
  REQUIRE_THAT(utils::bmax(0.0, std::numeric_limits<double>::max()),
               Catch::Matchers::WithinRel(std::numeric_limits<double>::max()));
  REQUIRE_THAT(utils::bmax(std::numeric_limits<double>::min(), 
                           std::numeric_limits<double>::max()),
               Catch::Matchers::WithinRel(std::numeric_limits<double>::max()));
  REQUIRE_THAT(utils::bmax(std::numeric_limits<double>::lowest(), 0.0),
               Catch::Matchers::WithinRel(0.0));
}

TEST_CASE("utils::bmin with extreme float values", "[utils::bmin]") {
  REQUIRE_THAT(utils::bmin(0.0, std::numeric_limits<double>::min()),
               Catch::Matchers::WithinRel(0.0));
  REQUIRE_THAT(utils::bmin(std::numeric_limits<double>::min(),
                           std::numeric_limits<double>::max()),
               Catch::Matchers::WithinRel(std::numeric_limits<double>::min()));
  REQUIRE_THAT(utils::bmin(std::numeric_limits<double>::lowest(), 0.0),
            Catch::Matchers::WithinRel(std::numeric_limits<double>::lowest()));
}
TEST_CASE("utils::bmax works with three ints", "[utils::bmax][int]") {
  REQUIRE(utils::bmax(1, 2, 3) == 3);
  REQUIRE(utils::bmax(2, 3, 1) == 3);
  REQUIRE(utils::bmax(3, 1, 2) == 3);
  REQUIRE(utils::bmax(-1, -2, -3) == -1);
}

TEST_CASE("utils::bmax works with three floats", "[utils::bmax][float]") {
  REQUIRE_THAT(utils::bmax(1.1f, 2.2f, 3.3f), 
    Catch::Matchers::WithinRel(3.3f));
  REQUIRE_THAT(utils::bmax(-1.1f, -2.2f, -3.3f), 
    Catch::Matchers::WithinRel(-1.1f));
}

TEST_CASE("utils::bmax works with three doubles", "[utils::bmax][double]") {
  REQUIRE_THAT(utils::bmax(1.1, 2.2, 3.3), 
    Catch::Matchers::WithinRel(3.3));
  REQUIRE_THAT(utils::bmax(-1.1, -2.2, -3.3), 
    Catch::Matchers::WithinRel(-1.1));
}

TEST_CASE("utils::bmin works with three ints", "[utils::bmin][int]") {
  REQUIRE(utils::bmin(1, 2, 3) == 1);
  REQUIRE(utils::bmin(2, 3, 1) == 1);
  REQUIRE(utils::bmin(3, 1, 2) == 1);
  REQUIRE(utils::bmin(-1, -2, -3) == -3);
}

TEST_CASE("utils::bmin works with three floats", "[utils::bmin][float]") {
  REQUIRE_THAT(utils::bmin(1.1f, 2.2f, 3.3f), 
    Catch::Matchers::WithinRel(1.1f));
  REQUIRE_THAT(utils::bmin(-1.1f, -2.2f, -3.3f), 
    Catch::Matchers::WithinRel(-3.3f));
}

TEST_CASE("utils::bmin works with three doubles", "[utils::bmin][double]") {
  REQUIRE_THAT(utils::bmin(1.1, 2.2, 3.3), 
    Catch::Matchers::WithinRel(1.1));
  REQUIRE_THAT(utils::bmin(-1.1, -2.2, -3.3), 
    Catch::Matchers::WithinRel(-3.3));
}

TEST_CASE("bfmod works with positive float values", "[bfmod][float]") {
  REQUIRE_THAT(utils::bfmod(5.5f, 2.0f), 
    Catch::Matchers::WithinAbs(1.5f, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(utils::bfmod(10.0f, 3.0f), 
    Catch::Matchers::WithinAbs(1.0f, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(utils::bfmod(9.0f, 4.5f), 
    Catch::Matchers::WithinAbs(0.0f, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(utils::bfmod(0.5f, 0.2f), 
    Catch::Matchers::WithinAbs(0.1f, std::numeric_limits<float>::epsilon()));
}

TEST_CASE("bfmod works with edge cases for float", "[bfmod][float]") {
  REQUIRE_THAT(utils::bfmod(1.0f, 1.0f), 
    Catch::Matchers::WithinAbs(0.0f, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(utils::bfmod(std::numeric_limits<float>::max(), 1.0f), 
    Catch::Matchers::WithinAbs(0.0f, std::numeric_limits<float>::epsilon()));
  REQUIRE(utils::bfmod(std::numeric_limits<float>::min(), 
    std::numeric_limits<float>::max()) == std::numeric_limits<float>::min());
}

TEST_CASE("bfmod works with positive double values", "[bfmod][double]") {
  REQUIRE_THAT(utils::bfmod(5.5, 2.0), 
    Catch::Matchers::WithinAbs(1.5, std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(utils::bfmod(10.0, 3.0), 
    Catch::Matchers::WithinAbs(1.0, std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(utils::bfmod(9.0, 4.5), 
    Catch::Matchers::WithinAbs(0.0, std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(utils::bfmod(0.5, 0.2), 
    Catch::Matchers::WithinAbs(0.1, std::numeric_limits<double>::epsilon()));
}

TEST_CASE("bfmod works with edge cases for double", "[bfmod][double]") {
  REQUIRE_THAT(utils::bfmod(1.0, 1.0), 
    Catch::Matchers::WithinAbs(0.0, std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(utils::bfmod(std::numeric_limits<double>::max(), 1.0), 
    Catch::Matchers::WithinAbs(0.0, std::numeric_limits<double>::epsilon()));
  REQUIRE(utils::bfmod(std::numeric_limits<double>::min(), 
    std::numeric_limits<double>::max()) == std::numeric_limits<double>::min());
}

struct SimpleStruct { int i; float f; };
class ComplexClass {
private:
    std::vector<int> data;
public:
    ComplexClass(std::initializer_list<int> list) : data(list) {}
    const std::vector<int>& getData() const { return data; }
};

TEST_CASE("swap works with primitive data types", "[swap]") {
    int a = 1, b = 2;
    utils::swap(a, b);
    REQUIRE(a == 2);
    REQUIRE(b == 1);

    float c = 1.5f, d = 2.5f;
    utils::swap(c, d);
    REQUIRE(c == 2.5f);
    REQUIRE(d == 1.5f);

    double e = 1.111, f = 2.222;
    utils::swap(e, f);
    REQUIRE(e == 2.222);
    REQUIRE(f == 1.111);

    char g = 'x', h = 'y';
    utils::swap(g, h);
    REQUIRE(g == 'y');
    REQUIRE(h == 'x');

    bool i = true, j = false;
    utils::swap(i, j);
    REQUIRE(i == false);
    REQUIRE(j == true);
}

#include <list>
TEST_CASE("swap works with std containers", "[swap]") {
    std::string s1 = "hello", s2 = "world";
    utils::swap(s1, s2);
    REQUIRE(s1 == "world");
    REQUIRE(s2 == "hello");

    std::vector<int> v1 = {1, 2, 3}, v2 = {4, 5, 6};
    utils::swap(v1, v2);
    REQUIRE(v1 == std::vector<int>({4, 5, 6}));
    REQUIRE(v2 == std::vector<int>({1, 2, 3}));

    std::array<int, 3> a1 = {7, 8, 9}, a2 = {10, 11, 12};
    utils::swap(a1, a2);
    REQUIRE(a1 == std::array<int, 3>({10, 11, 12}));
    REQUIRE(a2 == std::array<int, 3>({7, 8, 9}));

    std::list<int> l1 = {13, 14, 15}, l2 = {16, 17, 18};
    utils::swap(l1, l2);
    REQUIRE(l1 == std::list<int>({16, 17, 18}));
    REQUIRE(l2 == std::list<int>({13, 14, 15}));

    std::vector<std::string> vs1 = {"alpha", "beta"}, vs2 = {"gamma", "delta"};
    utils::swap(vs1, vs2);
    REQUIRE(vs1 == std::vector<std::string>({"gamma", "delta"}));
    REQUIRE(vs2 == std::vector<std::string>({"alpha", "beta"}));
}

TEST_CASE("swap works with custom structs and classes", "[swap]") {
    SimpleStruct ss1 = {1, 1.0f}, ss2 = {2, 2.0f};
    utils::swap(ss1, ss2);
    REQUIRE(ss1.i == 2);
    REQUIRE(ss1.f == 2.0f);
    REQUIRE(ss2.i == 1);
    REQUIRE(ss2.f == 1.0f);

    ComplexClass cc1 = {1, 2, 3}, cc2 = {4, 5, 6};
    utils::swap(cc1, cc2);
    REQUIRE(cc1.getData() == std::vector<int>({4, 5, 6}));
    REQUIRE(cc2.getData() == std::vector<int>({1, 2, 3}));
}
