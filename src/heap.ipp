///////////////////////////////////////////////////////////////////////////////
/// Heap Functions
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::swap(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
) {
  std::swap(heap_id[h0 + a], heap_id[h0 + b]);
  std::swap(heap_pq[h0 + a], heap_pq[h0 + b]);
}

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::maxheapify(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
) {
  while (true) {
    size_t largest = i;
    const size_t left = 2 * i + 1;
    const size_t right = 2 * i + 2;
    if ((left  < s) && (heap_pq[h0 + left] > heap_pq[h0 + largest])) {
      largest = left;
    }
    if ((right < s) && (heap_pq[h0 + right] > heap_pq[h0 + largest])) {
      largest = right;
    }
    if (i != largest) {
      swap(heap_id, heap_pq, h0, i, largest);
      i = largest;
      continue;
    }
    break;
  }
}

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::sort(
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const size_t h0,
  const size_t k
) {
  for (int i = k / 2 - 1; i >= 0; i--) {
    heap::maxheapify(heap_id, heap_pq, h0, k, i);
  }
  for (int i = k - 1; i >= 0; i--) {
    heap::swap(heap_id, heap_pq, h0, 0, i);
    heap::maxheapify(heap_id, heap_pq, h0, i, 0);
  }
}

/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/// SYCL Implementation
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::swap(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
) {
  const T1 tmpid = heap_id[h0 + a];
  heap_id[h0 + a] = heap_id[h0 + b];
  heap_id[h0 + b] = tmpid;

  const T2 tmppq = heap_pq[h0 + a];
  heap_pq[h0 + a] = heap_pq[h0 + b];
  heap_pq[h0 + b] = tmppq;
}

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::maxheapify(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t s, size_t i
) {
  while (true) {
    size_t largest = i;
    const size_t left = 2 * i + 1;
    const size_t right = 2 * i + 2;
    if ((left  < s) && 
        (heap_pq[h0 + left] > heap_pq[h0 + largest])) {
      largest = left;
    }
    if ((right < s) && 
        (heap_pq[h0 + right] > heap_pq[h0 + largest])) {
      largest = right;
    }
    if (i != largest) {
      swap(heap_id, heap_pq, h0, i, largest);
      i = largest;
      continue;
    }
    break;
  }
}

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
inline void heap::sort(
  device_accessor_readwrite_t<T1>& heap_id,
  device_accessor_readwrite_t<T2>& heap_pq,
  const size_t h0,
  const size_t k
) {
  for (int i = k / 2 - 1; i >= 0; i--) {
    heap::maxheapify(heap_id, heap_pq, h0, k, i);
  }
  for (int i = k - 1; i >= 0; i--) {
    swap(heap_id, heap_pq, h0, 0, i);
    maxheapify(heap_id, heap_pq, h0, i, 0);
  }
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
