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
 * @tparam Tf Numeric type of the point components.
 * @param p0 x-coordinate of the first point.
 * @param p1 y-coordinate of the first point.
 * @param p2 z-coordinate of the first point.
 * @param q0 x-coordinate of the second point.
 * @param q1 y-coordinate of the second point.
 * @param q2 z-coordinate of the second point.
 * @return The squared distance between the two points.
 */
template <typename Tf>
inline Tf get_distance(
  const Tf p0, const Tf p1, const Tf p2,
  const Tf q0, const Tf q1, const Tf q2
);

/**
 * @brief Sorts a set of 3D points into a grid of specified resolution.
 * 
 * The function sorts points into cells within a grid of `grid_resolution^3`
 * total cells, returning cell IDs and an offset for easy access to points in
 * each cell.
 * 
 * @tparam Ti Integer type for cell ID and offset values.
 * @tparam Tf Numeric type for point components.
 * @param xyzset Reference to a vector of 3D points to be sorted.
 * @param args Struct containing arguments such as `grid_resolution`.
 * @return A pair consisting of a vector of cell IDs and a vector of offsets
 * for accessing points in each cell.
 */
template <typename Ti, typename Tf>
const std::pair<std::vector<Ti>, std::vector<Ti>>
sort(std::vector<std::array<Tf, 3>>& xyzset, const args::xyzset& args);

/**
 * @brief Validates that all points in a set are within the range (0, 1).
 * 
 * @tparam Tf Numeric type for point components.
 * @param xyzset Vector of 3D points to be validated.
 * @return True if all points are within the specified range; otherwise, false.
 */
template <typename Tf>
bool 
validate_xyzset(const std::vector<std::array<Tf,3>>& xyzset);

/**
 * @brief Validates that a set of IDs is in ascending order.
 * 
 * @tparam Ti Type of the elements in the ID vector.
 * @param id Vector of IDs to be validated.
 * @return True if the IDs are in ascending order; otherwise, false.
 */
template <typename Ti>
bool 
validate_id(const std::vector<Ti>& id);

/**
 * @brief Placeholder function for validating offsets. Always returns true.
 * 
 * @tparam Ti Type of the elements in the offset vector.
 * @param offset Vector of offsets to be validated.
 * @return True, indicating offsets are considered valid.
 */
template <typename Ti>
bool 
validate_offset(const std::vector<Ti>& offset);

} // namespace xyzset

#include<xyzset.ipp>

#endif
