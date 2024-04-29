/**
 * @file heap_operations.hpp
 * @brief Defines heap manipulation functions for parallel computing
 * environments.
 */

#ifndef HEAP_HPP
#define HEAP_HPP

#include <libsycl.hpp>
/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */
namespace heap {

template <typename T1, typename T2>
inline void swap(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

template <typename T1, typename T2>
inline void maxheapify(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

template <typename T1, typename T2>
inline void sort(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */
/** 
 * @namespace heap
 * @brief Provides functions for managing heaps in a parallel computing
 * context.
 */
namespace heap {

/**
 * @brief Swaps two elements in both the heap of cell IDs and the heap of
 * priorities.
 * 
 * This operation affects both `heap_id` and `heap_pq`, ensuring the structure
 * of the heap is maintained across both.
 * 
 * @tparam T1 Type of elements in the heap of cell IDs.
 * @tparam T2 Type of elements in the heap of priorities.
 * @param heap_id Global heap of cell IDs with read-write access.
 * @param heap_pq Global heap of priorities/distances with read-write access.
 * @param h0 Offset to the local heap within the global heap.
 * @param a Index of the first element to swap.
 * @param b Index of the second element to swap.
 */
template <typename T1, typename T2>
inline void swap(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
);

/**
 * @brief Ensures the heap property is maintained from a given node down to its
 * descendants.
 * 
 * The function corrects the heap if the children of the node at index `i` are
 * greater than the node itself, ensuring the max-heap property is maintained.
 * 
 * @tparam T1 Type of elements in the heap of cell IDs.
 * @tparam T2 Type of elements in the heap of priorities.
 * @param heap_id Global heap of cell IDs with read-write access.
 * @param heap_pq Global heap of priorities/distances with read-write access.
 * @param h0 Offset to the local heap within the global heap.
 * @param s Size of the heap to consider for the operation.
 * @param i Index of the node to maxheapify from.
 */
template <typename T1, typename T2>
inline void maxheapify(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
);

/**
 * @brief Sorts elements in the heap using the heap sort algorithm.
 * 
 * This function sorts the global heaps `heap_id` and `heap_pq` in-place,
 * applying the heap sort algorithm to order the elements based on their
 * priority, while keeping the IDs synchronized.
 * 
 * @tparam T1 Type of elements in the heap of cell IDs.
 * @tparam T2 Type of elements in the heap of priorities.
 * @param heap_id Global heap of cell IDs with read-write access.
 * @param heap_pq Global heap of priorities/distances with read-write access.
 * @param h0 Offset to the local heap within the global heap.
 * @param k Number of elements to sort.
 */
template <typename T1, typename T2>
inline void sort(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t k
);

} // namespace heap
#include <heap.ipp>

#endif
