///////////////////////////////////////////////////////////////////////////////
/// Internal
///////////////////////////////////////////////////////////////////////////////
 
/* ------------------------------------------------------------------------- */

template <typename T>
static inline bool is_inshell(
  const int x, const int y, const int z,
  const int px, const int py, const int pz,
  const T r
) {
  return ((x > px - r && x < px + r) &&
          (y > py - r && y < py + r) &&
          (z > pz - r && z < pz + r));
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Cpu Implementation
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
void knni::compute(
  const T1 i,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<T1>& id,
  const std::vector<T1>& offset,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  std::vector<T1>& heap_id,
  std::vector<T2>& heap_pq,
  const struct args::knn& args
) {

  (void) xyzsize;
  (void) refsize;

  const T2 q0 = refset[i][0];
  const T2 q1 = refset[i][1];
  const T2 q2 = refset[i][2];

  const auto k = args.k;
  const auto gr = args.grid_resolution;
  const auto gl = 1.0f / args.grid_resolution;

  const T1 h0 = k * i;

  const int px = (id[i]) % gr;
  const int py = (id[i]  / gr) % gr;
  const int pz = (id[i]) / (gr * gr);
  
  const T2 gl2 = gl / 2;
  const T2 dx = std::fmod(q0, gl);
  const T2 dy = std::fmod(q1, gl);
  const T2 dz = std::fmod(q2, gl);
  const T2 min_dx = dx * (dx <= gl2) + (gl - dx) * (dx > gl2);
  const T2 min_dy = dy * (dy <= gl2) + (gl - dy) * (dy > gl2);
  const T2 min_dz = dz * (dz <= gl2) + (gl - dz) * (dz > gl2);
  const T2 min = std::min({min_dx, min_dy, min_dz});

  for (auto r = 0; r < gr; r++) {

    const int beg_z = std::max(pz - r, 0);
    const int end_z = std::min(pz + r, gr - 1);
    const int beg_y = std::max(py - r, 0);
    const int end_y = std::min(py + r, gr - 1);
    const int beg_x = std::max(px - r, 0);
    const int end_x = std::min(px + r, gr - 1);
    
    for (int z = beg_z; z <= end_z; z++) {
      for (int y = beg_y; y <= end_y; y++) {
        for (int x = beg_x; x <= end_x; x++) {
          if (is_inshell(x, y, z, px, py, pz, r)) {
            continue; 
          }

          const int cid = gr * gr * z + gr * y + x;
          const int offs0 = offset[cid];
          const int offs1 = offset[cid + 1];

          for (T1 p = offs0; p < offs1; p++) {

            const T2 p0 = xyzset[p][0];
            const T2 p1 = xyzset[p][1];
            const T2 p2 = xyzset[p][2];

            const T2 pq = xyzset::get_distance(p0, p1, p2, q0, q1, q2);
            if (pq < heap_pq[h0]) {
              heap_id[h0] = p;
              heap_pq[h0] = pq;
              heap::maxheapify<T1, T2>(heap_id, heap_pq, h0, k, 0);
            }
          }
        }
      }
    }
    if (heap_pq[h0] < utils::square(gl * (r - 1) + min)) {
      break; 
    }

  }

  heap::sort<T1,T2>(heap_id, heap_pq, h0, k);
  
  return;
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// Sycl Implementation
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2>
void knni::compute(
  const T1 i,
  const device_accessor_read_t<T2>& xyzset,
  const size_t xyzsize,
  const device_accessor_read_t<T1>& id,
  const device_accessor_read_t<T1>& offset,
  const device_accessor_read_t<T2>& refset,
  const size_t refsize,
  device_accessor_readwrite_t<T1> heap_id,
  device_accessor_readwrite_t<T2> heap_pq,
  const struct args::knn& args
) {

  const T2 q0 = refset[refsize * 0 + i];
  const T2 q1 = refset[refsize * 1 + i];
  const T2 q2 = refset[refsize * 2 + i];

  const auto k = args.k;
  const auto gr = args.grid_resolution;
  const auto gl = 1.0f / args.grid_resolution;

  const T1 h0 = k * i;

  const int px = (id[i]) % gr;
  const int py = (id[i]  / gr) % gr;
  const int pz = (id[i]) / (gr * gr);
  
  const T2 gl2 = gl / 2;
  const T2 dx = sycl::fmod(q0, gl);
  const T2 dy = sycl::fmod(q1, gl);
  const T2 dz = sycl::fmod(q2, gl);
  const T2 min_dx = dx * (dx <= gl2) + (gl - dx) * (dx > gl2);
  const T2 min_dy = dy * (dy <= gl2) + (gl - dy) * (dy > gl2);
  const T2 min_dz = dz * (dz <= gl2) + (gl - dz) * (dz > gl2);
  const T2 min = utils::bmin(min_dx, min_dy, min_dz);

  for (auto r = 0; r < gr; r++) {

    const int beg_z = utils::bmax(pz - r, 0);
    const int beg_y = utils::bmax(py - r, 0);
    const int beg_x = utils::bmax(px - r, 0);

    const int end_z = utils::bmin(pz + r, gr - 1);
    const int end_y = utils::bmin(py + r, gr - 1);
    const int end_x = utils::bmin(px + r, gr - 1);
    
    for (int z = beg_z; z <= end_z; z++) {
      for (int y = beg_y; y <= end_y; y++) {
        for (int x = beg_x; x <= end_x; x++) {
          if (is_inshell(x, y, z, px, py, pz, r)) {
            continue; 
          }

          const int cid = gr * gr * z + gr * y + x;
          const int offs0 = offset[cid];
          const int offs1 = offset[cid + 1];

          for (T1 p = offs0; p < offs1; p++) {
            if (p == i) {
              continue;
            }

            const T2 p0 = xyzset[xyzsize * 0 + p];
            const T2 p1 = xyzset[xyzsize * 1 + p];
            const T2 p2 = xyzset[xyzsize * 2 + p];

            const T2 pq = xyzset::get_distance(p0, p1, p2, q0, q1, q2);

            const bool cond = (pq < heap_pq[h0]);
            heap_id[h0] = !cond * heap_id[h0] + cond * p;
            heap_pq[h0] = !cond * heap_pq[h0] + cond * pq;
            heap::maxheapify<T1, T2>(heap_id, heap_pq, h0, cond * k, 0);
          }
        }
      }
    }

    // once again, this is problematic
    if (heap_pq[h0] < utils::square(gl * (r - 1) + min)) {
      break; 
    }
  }

  heap::sort<T1,T2>(heap_id, heap_pq, h0, k);
  return;
}
/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////

