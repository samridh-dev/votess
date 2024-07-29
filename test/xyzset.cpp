#include <catch2/catch_test_macros.hpp> 
#include <libsycl.hpp>
#include <votess.hpp>
#include <xyzset.hpp>

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <limits>

template <typename Ti, typename Tf>
static void test_xyzset(
  std::vector<std::array<Tf, 3>> xyzset,
  const unsigned int k, const unsigned int gr_max 
);

TEST_CASE("xyzset regression 1: standard", "[xyzset]") {
  
  std::vector<std::array<float, 3>> xyzset = {
    {0.605223f, 0.108484f, 0.090937f}, {0.500792f, 0.499641f, 0.464576f},
    {0.437936f, 0.786332f, 0.160392f}, {0.663354f, 0.170894f, 0.810284f},
    {0.614869f, 0.096867f, 0.204147f}, {0.556911f, 0.895342f, 0.802266f},
    {0.305748f, 0.124146f, 0.516249f}, {0.406888f, 0.157835f, 0.919622f},
    {0.094412f, 0.861991f, 0.798644f}, {0.511958f, 0.560537f, 0.345479f}
  };

  const unsigned short int k = 1;
  const unsigned short int gr_max = 16;
  test_xyzset<int, float>(xyzset, k, gr_max);

}

TEST_CASE("xyzset regression 2: small fibonacci sphere", "[xyzset]") {

  std::vector<std::array<float, 3>> xyzset = {
    {0.500000f, 0.750000f, 0.500000f}, {0.476957f, 0.748039f, 0.521109f},
    {0.503856f, 0.746078f, 0.456062f}, {0.532803f, 0.744118f, 0.542785f},
    {0.438820f, 0.742157f, 0.489178f}, {0.558493f, 0.740196f, 0.462792f},
    {0.480325f, 0.738235f, 0.573190f}, {0.462345f, 0.736275f, 0.427498f},
    {0.581872f, 0.734314f, 0.529900f}, {0.414719f, 0.732353f, 0.535203f},
    {0.541136f, 0.730392f, 0.412095f}, {0.530402f, 0.728431f, 0.596927f},
    {0.408389f, 0.726471f, 0.446910f}, {0.607414f, 0.724510f, 0.476385f},
    {0.434496f, 0.722549f, 0.593173f}, {0.484881f, 0.720588f, 0.383328f},
    {0.592715f, 0.718627f, 0.578140f}, {0.375385f, 0.716667f, 0.505153f},
    {0.590778f, 0.714706f, 0.409663f}, {0.493935f, 0.712745f, 0.631159f},
    {0.413873f, 0.710784f, 0.396791f}, {0.636223f, 0.708824f, 0.518329f},
    {0.384764f, 0.706863f, 0.580178f}, {0.531437f, 0.704902f, 0.360261f},
    {0.572586f, 0.702941f, 0.626673f}, {0.358350f, 0.700980f, 0.454810f},
    {0.637347f, 0.699020f, 0.436542f}, {0.440607f, 0.697059f, 0.641917f},
    {0.447092f, 0.695098f, 0.352903f}, {0.640514f, 0.693137f, 0.573850f},
    {0.344226f, 0.691176f, 0.541062f}, {0.588369f, 0.689216f, 0.362565f},
    {0.528055f, 0.687255f, 0.663244f}, {0.367312f, 0.685294f, 0.397239f},
    {0.669387f, 0.683333f, 0.485967f}, {0.383159f, 0.681373f, 0.626302f},
    {0.500849f, 0.679412f, 0.325900f}, {0.618319f, 0.677451f, 0.630429f},
    {0.322706f, 0.675490f, 0.483568f}, {0.643353f, 0.673529f, 0.391200f},
    {0.467454f, 0.671569f, 0.678899f}, {0.402178f, 0.669608f, 0.344552f},
    {0.678862f, 0.667647f, 0.549019f}, {0.333441f, 0.665686f, 0.585475f},
    {0.565675f, 0.663725f, 0.322854f}, {0.571868f, 0.661765f, 0.676542f},
    {0.326266f, 0.659804f, 0.417664f}, {0.685265f, 0.657843f, 0.442881f},
    {0.401176f, 0.655882f, 0.668625f}, {0.458640f, 0.653922f, 0.307393f},
    {0.661838f, 0.651961f, 0.614962f}, {0.301535f, 0.650000f, 0.524734f},
    {0.630577f, 0.648039f, 0.346592f}, {0.507394f, 0.646078f, 0.702747f},
    {0.356613f, 0.644118f, 0.354499f}, {0.705381f, 0.642157f, 0.510501f},
    {0.340431f, 0.640196f, 0.631844f}, {0.528784f, 0.638235f, 0.293693f},
    {0.618865f, 0.636275f, 0.672628f}, {0.294515f, 0.634314f, 0.452718f},
    {0.684533f, 0.632353f, 0.395451f}, {0.434179f, 0.630392f, 0.702893f},
    {0.410990f, 0.628431f, 0.304853f}, {0.698523f, 0.626471f, 0.584224f},
    {0.295650f, 0.624510f, 0.572377f}, {0.602315f, 0.622549f, 0.307611f},
    {0.554786f, 0.620588f, 0.712031f}, {0.315482f, 0.618627f, 0.380080f},
    {0.718094f, 0.616667f, 0.463613f}, {0.363133f, 0.614706f, 0.674957f},
    {0.482662f, 0.612745f, 0.277541f}, {0.663769f, 0.610784f, 0.652992f},
    {0.274939f, 0.608824f, 0.497804f}, {0.668134f, 0.606863f, 0.348966f},
    {0.477956f, 0.604902f, 0.725853f}, {0.363154f, 0.602941f, 0.317856f},
    {0.724803f, 0.600980f, 0.542030f}, {0.305120f, 0.599020f, 0.621313f},
    {0.561978f, 0.597059f, 0.278103f}, {0.604558f, 0.595098f, 0.706214f},
    {0.282861f, 0.593137f, 0.418292f}, {0.716026f, 0.591176f, 0.413285f},
    {0.398956f, 0.589216f, 0.710549f}, {0.432068f, 0.587255f, 0.275786f},
    {0.702165f, 0.585294f, 0.619809f}, {0.269312f, 0.583333f, 0.548361f},
    {0.637833f, 0.581373f, 0.307957f}, {0.528169f, 0.579412f, 0.735373f},
    {0.319746f, 0.577451f, 0.345049f}, {0.738211f, 0.575490f, 0.492476f},
    {0.328994f, 0.573529f, 0.666885f}, {0.513399f, 0.571569f, 0.260838f},
    {0.652037f, 0.569608f, 0.685848f}, {0.261800f, 0.567647f, 0.465581f},
    {0.699339f, 0.565686f, 0.364172f}, {0.444644f, 0.563725f, 0.735318f},
    {0.381614f, 0.561765f, 0.288647f}, {0.730527f, 0.559804f, 0.576030f},
    {0.278226f, 0.557843f, 0.599851f}, {0.596262f, 0.555882f, 0.276146f},
    {0.580376f, 0.553922f, 0.730504f}, {0.284656f, 0.551961f, 0.384126f},
    {0.737456f, 0.550000f, 0.439879f}, {0.365303f, 0.548039f, 0.705059f},
    {0.460746f, 0.546078f, 0.257439f}, {0.693074f, 0.544118f, 0.652564f},
    {0.254235f, 0.542157f, 0.517949f}, {0.669320f, 0.540196f, 0.320515f},
    {0.496384f, 0.538235f, 0.747032f}, {0.335602f, 0.536275f, 0.315183f},
    {0.746343f, 0.534314f, 0.525257f}, {0.301082f, 0.532353f, 0.647935f},
    {0.546794f, 0.530392f, 0.256306f}, {0.630230f, 0.528431f, 0.711499f},
    {0.260899f, 0.526471f, 0.431957f}, {0.722448f, 0.524510f, 0.388572f},
    {0.411176f, 0.522549f, 0.732598f}, {0.408315f, 0.520588f, 0.268332f},
    {0.724234f, 0.518627f, 0.608960f}, {0.260923f, 0.516667f, 0.571163f},
    {0.628281f, 0.514706f, 0.285926f}, {0.550036f, 0.512745f, 0.744610f},
    {0.297799f, 0.510784f, 0.353377f}, {0.748216f, 0.508824f, 0.471522f},
    {0.336170f, 0.506863f, 0.688713f}, {0.493328f, 0.504902f, 0.250137f},
    {0.673721f, 0.502941f, 0.679757f}, {0.250465f, 0.500980f, 0.484799f},
    {0.694268f, 0.499020f, 0.342650f}, {0.463043f, 0.497059f, 0.747236f},
    {0.360262f, 0.495098f, 0.292758f}, {0.742983f, 0.493137f, 0.558412f},
    {0.281429f, 0.491176f, 0.621032f}, {0.579387f, 0.489216f, 0.263185f},
    {0.601391f, 0.487255f, 0.728161f}, {0.271216f, 0.485294f, 0.400293f},
    {0.735933f, 0.483333f, 0.419019f}, {0.380798f, 0.481373f, 0.718961f},
    {0.440026f, 0.479412f, 0.258175f}, {0.707432f, 0.477451f, 0.637709f},
    {0.254209f, 0.475490f, 0.538547f}, {0.655075f, 0.473529f, 0.305704f},
    {0.516883f, 0.471569f, 0.747804f}, {0.320330f, 0.469608f, 0.328842f},
    {0.747851f, 0.467647f, 0.504837f}, {0.314173f, 0.465686f, 0.663680f},
    {0.526429f, 0.463725f, 0.254062f}, {0.646466f, 0.461765f, 0.698962f},
    {0.257909f, 0.459804f, 0.452288f}, {0.710459f, 0.457843f, 0.371822f},
    {0.431493f, 0.455882f, 0.736348f}, {0.391024f, 0.453922f, 0.279771f},
    {0.728768f, 0.451961f, 0.588642f}, {0.271802f, 0.450000f, 0.589026f},
    {0.607949f, 0.448039f, 0.280576f}, {0.568501f, 0.446078f, 0.734308f},
    {0.291594f, 0.444118f, 0.373729f}, {0.738517f, 0.442157f, 0.452422f},
    {0.356544f, 0.440196f, 0.695816f}, {0.473562f, 0.438235f, 0.259197f},
    {0.681773f, 0.436275f, 0.659367f}, {0.258841f, 0.434314f, 0.505262f},
    {0.673877f, 0.432353f, 0.333595f}, {0.484231f, 0.430392f, 0.739596f},
    {0.350144f, 0.428431f, 0.313128f}, {0.736142f, 0.426471f, 0.536477f},
    {0.301746f, 0.424510f, 0.632275f}, {0.556686f, 0.422549f, 0.269158f},
    {0.613822f, 0.420588f, 0.707938f}, {0.276241f, 0.418627f, 0.423773f},
    {0.715857f, 0.416667f, 0.405335f}, {0.405063f, 0.414706f, 0.714970f},
    {0.425024f, 0.412745f, 0.278042f}, {0.704567f, 0.410784f, 0.612663f},
    {0.273793f, 0.408824f, 0.554930f}, {0.629260f, 0.406863f, 0.307341f},
    {0.534707f, 0.404902f, 0.728586f}, {0.320635f, 0.402941f, 0.355404f},
    {0.729097f, 0.400980f, 0.485514f}, {0.341450f, 0.399020f, 0.664817f},
    {0.505556f, 0.397059f, 0.272245f}, {0.649160f, 0.395098f, 0.671017f},
    {0.275405f, 0.393137f, 0.474756f}, {0.681904f, 0.391176f, 0.367454f},
    {0.455593f, 0.389216f, 0.719670f}, {0.384865f, 0.387255f, 0.308865f},
    {0.713046f, 0.385294f, 0.562881f}, {0.301351f, 0.383333f, 0.597094f},
    {0.580509f, 0.381373f, 0.295193f}, {0.578597f, 0.379412f, 0.704404f},
    {0.304949f, 0.377451f, 0.402857f}, {0.708373f, 0.375490f, 0.440183f},
    {0.387353f, 0.373529f, 0.683891f}, {0.459068f, 0.371569f, 0.289453f},
    {0.671451f, 0.369608f, 0.626896f}, {0.289065f, 0.367647f, 0.522118f},
    {0.639777f, 0.365686f, 0.342132f}, {0.503550f, 0.363725f, 0.709563f},
    {0.356711f, 0.361765f, 0.348808f}, {0.706475f, 0.359804f, 0.514601f},
    {0.338940f, 0.357843f, 0.627871f}, {0.532170f, 0.355882f, 0.298269f},
    {0.611776f, 0.353922f, 0.669314f}, {0.304593f, 0.351961f, 0.451004f},
    {0.675903f, 0.350000f, 0.404826f}, {0.435070f, 0.348039f, 0.687595f},
    {0.421761f, 0.346078f, 0.319205f}, {0.678403f, 0.344118f, 0.579830f},
    {0.316026f, 0.342157f, 0.561147f}, {0.593568f, 0.340196f, 0.332048f},
    {0.544075f, 0.338235f, 0.685444f}, {0.343627f, 0.336275f, 0.393975f},
    {0.685225f, 0.334314f, 0.472800f}, {0.382902f, 0.332353f, 0.643814f},
    {0.489304f, 0.330392f, 0.316646f}, {0.630428f, 0.328431f, 0.626700f},
    {0.320112f, 0.326471f, 0.494733f}, {0.634755f, 0.324510f, 0.383620f},
    {0.479477f, 0.322549f, 0.674900f}, {0.398159f, 0.320588f, 0.358792f},
    {0.668478f, 0.318627f, 0.534915f}, {0.353981f, 0.316667f, 0.586990f},
    {0.548293f, 0.314706f, 0.339271f}, {0.572009f, 0.312745f, 0.649166f},
    {0.348227f, 0.310784f, 0.439482f}, {0.650642f, 0.308824f, 0.442917f},
    {0.428540f, 0.306863f, 0.641745f}, {0.457600f, 0.304902f, 0.349537f},
    {0.630793f, 0.302941f, 0.581001f}, {0.351343f, 0.300980f, 0.528148f},
    {0.589035f, 0.299020f, 0.380922f}, {0.514517f, 0.297059f, 0.645273f},
    {0.393227f, 0.295098f, 0.404529f}, {0.640374f, 0.293137f, 0.498307f},
    {0.399775f, 0.291176f, 0.594062f}, {0.510136f, 0.289216f, 0.365958f},
    {0.581138f, 0.287255f, 0.603228f}, {0.373630f, 0.285294f, 0.479215f},
    {0.604420f, 0.283333f, 0.431794f}, {0.469937f, 0.281373f, 0.617466f},
    {0.444521f, 0.279412f, 0.396256f}, {0.607447f, 0.277451f, 0.537777f},
    {0.398855f, 0.275490f, 0.543186f}, {0.543718f, 0.273529f, 0.403564f},
    {0.531569f, 0.271569f, 0.596553f}, {0.415447f, 0.269608f, 0.452353f},
    {0.589864f, 0.267647f, 0.479105f}, {0.450735f, 0.265686f, 0.571903f},
    {0.488525f, 0.263725f, 0.419113f}, {0.558537f, 0.261765f, 0.548139f},
    {0.430775f, 0.259804f, 0.503699f}, {0.543507f, 0.257843f, 0.455647f},
    {0.498159f, 0.255882f, 0.553881f}, {0.471334f, 0.253922f, 0.466479f},
    {0.531019f, 0.251961f, 0.503793f}, {0.500000f, 0.250000f, 0.500000f}
  };
  
  const unsigned short int k = 1;
  const unsigned short int gr_max = 16;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

template <typename Tf>
static std::vector<std::array<Tf, 3>> 
create_fibonacci(const size_t n) {
  std::vector<std::array<Tf, 3>> points;
  Tf phi = M_PI * (3. - sqrt(5.)); 
  Tf scale = 0.25; 
  Tf center_x = 0.5, center_y = 0.5, center_z = 0.5; 
  for (size_t i = 0; i < n; ++i) {
    Tf y = (1 - (i / Tf(n - 1)) * 2) * scale + center_y; 
    Tf r = sqrt(1 - pow((y - center_y) / scale, 2)); 
    Tf theta = phi * i;
    Tf x = cos(theta) * r * scale + center_x; 
    Tf z = sin(theta) * r * scale + center_z; 
    points.push_back({x, y, z});
  }
  return points;
}

TEST_CASE("xyzset regression 3: Large Fibonacci Sphere - float", "[xyzset]") {
  const size_t size = 1000000;
  auto xyzset = create_fibonacci<float>(size);
  const unsigned short int k = 1;
  const unsigned short int gr_max = 32;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 3: Large Fibonacci Sphere - double", "[xyzset]") {
  const size_t size = 1000000;
  auto xyzset = create_fibonacci<double>(size);
  const unsigned short int k = 1;
  const unsigned short int gr_max = 32;
  test_xyzset<int, double>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 5: Empty Input - float", "[xyzset]") {
  std::vector<std::array<float, 3>> xyzset = {};
  const unsigned short int k = 1;
  const unsigned short int gr_max = 0;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 5: Empty Input - double", "[xyzset]") {
  std::vector<std::array<double, 3>> xyzset = {};
  const unsigned short int k = 1;
  const unsigned short int gr_max = 0;
  test_xyzset<int, double>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 6: Single Point - float", "[xyzset]") {
  std::vector<std::array<float, 3>> xyzset = {{0.5f, 0.5f, 0.5f}};
  const unsigned short int k = 1;
  const unsigned short int gr_max = 0;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 6: Single Point - double", "[xyzset]") {
  std::vector<std::array<double, 3>> xyzset = {{0.5f, 0.5f, 0.5f}};
  const unsigned short int k = 1;
  const unsigned short int gr_max = 0;
  test_xyzset<int, double>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 7: Boundary Values - float", "[xyzset]") {
  const float eps = std::numeric_limits<float>::epsilon();
  std::vector<std::array<float, 3>> xyzset = {
    {eps, eps, eps}, {1.0f - eps, 1.0f - eps, 1.0f - eps},
    {eps, eps, 1.0f - eps}, {1.0f - eps, 1.0f - eps, eps},
    {eps, 1.0f - eps, eps}, {1.0f - eps, eps, 1.0f - eps}
  };
  const unsigned short int k = 1;
  const unsigned short int gr_max = 1;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 7: Boundary Values - double", "[xyzset]") {
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<std::array<double, 3>> xyzset = {
    {eps, eps, eps}, {1.0 - eps, 1.0 - eps, 1.0 - eps},
    {eps, eps, 1.0 - eps}, {1.0 - eps, 1.0 - eps, eps},
    {eps, 1.0 - eps, eps}, {1.0 - eps, eps, 1.0 - eps}
  };
  const unsigned short int k = 1;
  const unsigned short int gr_max = 1;
  test_xyzset<int, double>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 8: Random Points - float", "[xyzset]") {
  std::vector<std::array<float, 3>> xyzset;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (size_t i = 0; i < 1000; ++i) {
    xyzset.push_back({dis(gen), dis(gen), dis(gen)});
  }

  const unsigned short int k = 1;
  const unsigned short int gr_max = 16;
  test_xyzset<int, float>(xyzset, k, gr_max);
}

TEST_CASE("xyzset regression 8: Random Points - double", "[xyzset]") {
  std::vector<std::array<double, 3>> xyzset;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (size_t i = 0; i < 1000; ++i) {
    xyzset.push_back({dis(gen), dis(gen), dis(gen)});
  }

  const unsigned short int k = 1;
  const unsigned short int gr_max = 16;
  test_xyzset<int, double>(xyzset, k, gr_max);
}

///////////////////////////////////////////////////////////////////////////////

#define TEST_XYZSET_USE_ALTER 0
template <typename Ti, typename Tf>
static void test_xyzset(
  std::vector<std::array<Tf, 3>> xyzset,
  const unsigned int k, const unsigned int gr_max
) {
  
  std::vector<std::array<Tf, 3>> refset = xyzset;

  for (size_t gr = 0; gr <= gr_max; gr++) {

    SECTION("case: gr = " + std::to_string(gr)) {
      votess::vtargs args(k, gr);
      auto [id, offset] = xyzset::sort<Ti, Tf>(xyzset, args.xyzset);

      REQUIRE(xyzset::validate_xyzset<Tf>(xyzset) == true);
      REQUIRE(xyzset::validate_id<Ti>(id) == true);
      REQUIRE(xyzset::validate_offset<Ti>(offset) == true);
      REQUIRE(xyzset::validate_sort<Ti, Tf>(xyzset, id, gr) == true);
    
#if TEST_XYZSET_USE_ALTER
      for (const auto& element : refset) {
        auto found = std::find(xyzset.begin(), xyzset.end(), element);
        REQUIRE(found != xyzset.end());
      }
#endif

    }
  }
}
