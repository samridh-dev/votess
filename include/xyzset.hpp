/**
 * @file xyzset.hpp
 * @brief Provides functionality for sorting 3D point sets and validating their
 * properties.
 */

#ifndef XYZSET_HPP
#define XYZSET_HPP

#include <algorithm>
#include <libsycl.hpp>
#include <arguments.hpp>

/** 
 * @namespace xyzset
 * @brief Encapsulates functions for sorting and validating 3D point sets.
 */
namespace xyzset {

/**
 * @brief Calculates the squared distance between two 3D points.
 * 
 * @tparam T2 Numeric type of the point components.
 * @param p0 x-coordinate of the first point.
 * @param p1 y-coordinate of the first point.
 * @param p2 z-coordinate of the first point.
 * @param q0 x-coordinate of the second point.
 * @param q1 y-coordinate of the second point.
 * @param q2 z-coordinate of the second point.
 * @return The squared distance between the two points.
 */
template <typename T2>
inline T2 get_distance(
  const T2 p0, const T2 p1, const T2 p2,
  const T2 q0, const T2 q1, const T2 q2
);

/**
 * @brief Sorts a set of 3D points into a grid of specified resolution.
 * 
 * The function sorts points into cells within a grid of `grid_resolution^3`
 * total cells, returning cell IDs and an offset for easy access to points in
 * each cell.
 * 
 * @tparam T1 Integer type for cell ID and offset values.
 * @tparam T2 Numeric type for point components.
 * @param xyzset Reference to a vector of 3D points to be sorted.
 * @param args Struct containing arguments such as `grid_resolution`.
 * @return A pair consisting of a vector of cell IDs and a vector of offsets
 * for accessing points in each cell.
 */
template <typename T1, typename T2>
const std::pair<std::vector<T1>, std::vector<T1>>
sort(std::vector<std::array<T2, 3>>& xyzset, const args::xyzset& args);

/**
 * @brief Validates that all points in a set are within the range (0, 1).
 * 
 * @tparam T2 Numeric type for point components.
 * @param xyzset Vector of 3D points to be validated.
 * @return True if all points are within the specified range; otherwise, false.
 */
template <typename T2>
bool 
validate_xyzset(const std::vector<std::array<T2,3>>& xyzset);

/**
 * @brief Validates that a set of IDs is in ascending order.
 * 
 * @tparam T1 Type of the elements in the ID vector.
 * @param id Vector of IDs to be validated.
 * @return True if the IDs are in ascending order; otherwise, false.
 */
template <typename T1>
bool 
validate_id(const std::vector<T1>& id);

/**
 * @brief Placeholder function for validating offsets. Always returns true.
 * 
 * @tparam T1 Type of the elements in the offset vector.
 * @param offset Vector of offsets to be validated.
 * @return True, indicating offsets are considered valid.
 */
template <typename T1>
bool 
validate_offset(const std::vector<T1>& offset);

} // namespace xyzset

#include<xyzset.ipp>

#endif
