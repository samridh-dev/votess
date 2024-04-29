#include <algorithm>
#include <libsycl.hpp>

#include <utils.hpp>
namespace xyzset {

template <typename T2>
inline T2 get_distance(
  const T2 p0, const T2 p1, const T2 p2,
  const T2 q0, const T2 q1, const T2 q2
) {
  using namespace utils;
  return square(p0 - q0) + square(p1 - q1) + square(p2 - q2);
}

template <typename T1, typename T2>
const std::pair<std::vector<T1>, std::vector<T1>>
sort(std::vector<std::array<T2,3>>& xyzset, const args::xyzset& args) {

  const auto gr = args.grid_resolution;
  const T2 gl = 1.0f / gr; // TODO: make this template safe
  const T1 idmax = gr * gr * gr;

  std::vector<T1> id(xyzset.size());
  std::vector<T1> offset(0,0);
  
  // init
  for (size_t i = 0; i < xyzset.size(); i++) {
    id[i] = static_cast<T1>(std::floor(xyzset[i][0] / gl))
          + static_cast<T1>(std::floor(xyzset[i][1] / gl)) * gr  
          + static_cast<T1>(std::floor(xyzset[i][2] / gl)) * gr * gr; 
  };
  
  // sort
  const int bits = sizeof(T1) * 8;
  const size_t size = id.size();
  std::vector<T1> tmp_id(size);
  std::vector<std::array<T2,3>> tmp_xyzset(size);
  for (int shift = 0; shift < bits; shift += 8) {
    size_t count[256] = {0};
    for (size_t i = 0; i < size; i++) {
      count[(id[i] >> shift) & 0xFF]++;
    }
    for (int i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }
    for (int i = size - 1; i >= 0; i--) {
      const size_t idx = (id[i] >> shift) & 0xFF;
      const size_t sortedIdx = --count[idx];
      tmp_id[sortedIdx] = id[i];
      tmp_xyzset[sortedIdx] = xyzset[i];
    }
    std::copy(tmp_id.begin(), tmp_id.end(), id.begin());
    std::copy(tmp_xyzset.begin(), tmp_xyzset.end(), xyzset.begin());
  }
  
  // update
  offset.resize(idmax + 1, 0);
  for (size_t i = 0; i < id.size(); i++) {
    offset[id[i] + 1]++;
  }
  T1 update = 0;
  for (size_t i = 0; i < offset.size(); i++) {
    update += offset[i];
    offset[i] = update;
  }
  offset[offset.size() - 1] = id.size();

  return std::make_pair(id, offset);
}

template <typename T2>
bool
validate_xyzset(const std::vector<std::array<T2,3>>& xyzset) {
  for (size_t i = 0; i < xyzset.size(); i++) {
    for (size_t j = 0; j < xyzset[0].size(); j++) {
      if (xyzset[i][j] >= 1.0f || xyzset[i][j] <= 0.0f) {
        return false;
      }
    }
  }
  return true;
}

template <typename T1>
bool
validate_id(const std::vector<T1>& id) {
  for (size_t i = 1; i < id.size(); i++) {
    if (id[i] < id[i - 1]) {
      return false;
    }
  }
  return true;
}

template <typename T1>
bool
validate_offset(const std::vector<T1>& offset) {
  (void) offset;
  return true;
}

template <typename T1, typename T2>
bool validate_sort(
  const std::vector<std::array<T2, 3>>& xyzset,
  const std::vector<T1>& id,
  const T1 gr
) {
  T2 gl = 1.0f / gr;
  for (size_t i = 0; i < xyzset.size(); ++i) {
    T1 cid = static_cast<T1>(std::floor(xyzset[i][0] / gl)) +
             static_cast<T1>(std::floor(xyzset[i][1] / gl) * gr) +
             static_cast<T1>(std::floor(xyzset[i][2] / gl) * gr * gr);
    if (cid != id[i]) {
      return false;
    }
  }
  return true;
}

}
