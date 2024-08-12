///////////////////////////////////////////////////////////////////////////////
/// Heap Functions
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */
/// CPU Implementation
/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
inline void heap::swap(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
) {
  std::swap(heap_id[h0 + a], heap_id[h0 + b]);
  std::swap(heap_pq[h0 + a], heap_pq[h0 + b]);
}

/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
inline void heap::maxheapify(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
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

template <typename Ti, typename Tf>
inline void heap::sort(
  std::vector<Ti>& heap_id,
  std::vector<Tf>& heap_pq,
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

template <typename Ti, typename Tf>
inline void heap::swap(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const size_t h0,
  const size_t a, const size_t b
) {
  const Ti tmpid = heap_id[h0 + a];
  heap_id[h0 + a] = heap_id[h0 + b];
  heap_id[h0 + b] = tmpid;

  const Tf tmppq = heap_pq[h0 + a];
  heap_pq[h0 + a] = heap_pq[h0 + b];
  heap_pq[h0 + b] = tmppq;
}

/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
inline void heap::maxheapify(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
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

template <typename Ti, typename Tf>
inline void heap::sort(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
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
/// New IMplemenetation
/* ------------------------------------------------------------------------- */

// terminologies:
//   's' : row size
//   'i' : column index

template <typename Ti, typename Tf>
inline void heap::swap(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int a, const int b
) {
  const Ti tmpid = heap_id[s * a + i];
  heap_id[s * a + i] = heap_id[s * b + i];
  heap_id[s * b + i] = tmpid;

  const Tf tmppq = heap_pq[s * a + i];
  heap_pq[s * a + i] = heap_pq[s * b + i];
  heap_pq[s * b + i] = tmppq;
}

/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
inline void heap::maxheapify(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int k, int idx
) {
#if 1
  while (true) {
    int largest = idx;
    const int left = 2 * idx + 1;
    const int right = 2 * idx + 2;
    if ((left < k) && 
        (heap_pq[s * left + i] > heap_pq[s * largest + i])) {
      largest = left;
    }
    if ((right < k) && 
        (heap_pq[s * right + i] > heap_pq[s * largest + i])) {
      largest = right;
    }
    if (idx != largest) {
      swap(heap_id, heap_pq, s, i, idx, largest);
      idx = largest;
      continue;
    }
    break;
  }
#else
  while (true) {
    int largest = idx;
    const int left = 2 * idx + 1;
    const int right = 2 * idx + 2;

    if (left < k) {
      const int sl = static_cast<int>(heap_pq[s * left + i] > heap_pq[s * largest + i]);
      largest = (!sl) * largest + sl * left;
    }

    if (right < k) {
      const int sr = static_cast<int>(heap_pq[s * right + i] > heap_pq[s * largest + i]);
      largest = (!sr) * largest + sr * right;
    }

    if (idx == largest) break;

    swap(heap_id, heap_pq, s, i, idx, largest);
    idx = largest;
  }
#endif

}

template <typename Ti, typename Tf>
inline void heap::sort(
  const device_accessor_readwrite_t<Ti>& heap_id,
  const device_accessor_readwrite_t<Tf>& heap_pq,
  const int s, const int i,
  const int k
) {
  for (int idx = k / 2 - 1; idx >= 0; idx--) {
    heap::maxheapify(heap_id, heap_pq, s, i, k, idx);
  }
  for (int idx = k - 1; idx >= 0; idx--) {
    swap(heap_id, heap_pq, s, i, 0, idx);
    maxheapify(heap_id, heap_pq, s, i, idx, 0);
  }
}

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
