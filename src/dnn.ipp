///////////////////////////////////////////////////////////////////////////////
/// CPU Implementation
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2, typename T3>
void dnni::compute(
  const T1 i,
  std::vector<cc::state>& states,
  T2* P,
  T3* T,
  T3* dR,
  std::vector<T1>& knn,
  std::vector<T1>& dknn,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  const struct args::cc& args
) {

  (void) xyzsize;
  (void) refsize;
  
  // TODO : make this swappable for future
  static const T2 p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const T3 t_init[] = { 
    2,5,0, 5,3,0, 1,5,2, 5,1,3, 
    4,2,0, 4,0,3, 2,4,1, 4,3,1 
  };

  const unsigned short int p_initsize = sizeof(p_init) / (sizeof(*p_init) * 4);
  const unsigned short int t_initsize = sizeof(t_init) / (sizeof(*t_init) * 3);

  const unsigned short int k = args.k;
  const unsigned short int p_maxsize = args.p_maxsize;
  const unsigned short int t_maxsize = args.t_maxsize;

  unsigned short int p_size = p_initsize;
  unsigned short int t_size = t_initsize; 
  unsigned short int r_size = 0; 

  cc::state& state = states[i];

  T2 vertex[4];
  T2 bisector[4];

  const T2 px = refset[i][0];
  const T2 py = refset[i][1];
  const T2 pz = refset[i][2];
 
  for (size_t j = 0; j < p_initsize; j++) {
    P[4 * p_maxsize * i + 4 * j + 0] = p_init[4 * j + 0];
    P[4 * p_maxsize * i + 4 * j + 1] = p_init[4 * j + 1];
    P[4 * p_maxsize * i + 4 * j + 2] = p_init[4 * j + 2];
    P[4 * p_maxsize * i + 4 * j + 3] = p_init[4 * j + 3];
  }

  for (size_t j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }
  
  for (size_t neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[k * i + neighbor];

    const T2 qx = xyzset[q][0];
    const T2 qy = xyzset[q][1];
    const T2 qz = xyzset[q][2];
  
    r_size = 0;
    T2 sradius = 0.00f;
  
    planes::bisect<T2>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const T3& t0 = T[3 * t_maxsize * i + 3 * t_index + 0];
      const T3& t1 = T[3 * t_maxsize * i + 3 * t_index + 1];
      const T3& t2 = T[3 * t_maxsize * i + 3 * t_index + 2];
  
      const T2& plane_00 = P[4 * p_maxsize * i + 4 * t0 + 0];
      const T2& plane_01 = P[4 * p_maxsize * i + 4 * t0 + 1];
      const T2& plane_02 = P[4 * p_maxsize * i + 4 * t0 + 2];
      const T2& plane_03 = P[4 * p_maxsize * i + 4 * t0 + 3];

      const T2& plane_10 = P[4 * p_maxsize * i + 4 * t1 + 0];
      const T2& plane_11 = P[4 * p_maxsize * i + 4 * t1 + 1];
      const T2& plane_12 = P[4 * p_maxsize * i + 4 * t1 + 2];
      const T2& plane_13 = P[4 * p_maxsize * i + 4 * t1 + 3];

      const T2& plane_20 = P[4 * p_maxsize * i + 4 * t2 + 0];
      const T2& plane_21 = P[4 * p_maxsize * i + 4 * t2 + 1];
      const T2& plane_22 = P[4 * p_maxsize * i + 4 * t2 + 2];
      const T2& plane_23 = P[4 * p_maxsize * i + 4 * t2 + 3];

      // TODO : implement exception handling
      planes::intersect<T2>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const T2& b0 = bisector[0];
      const T2& b1 = bisector[1];
      const T2& b2 = bisector[2];
      const T2& b3 = bisector[3];
  
      const T2& v0 = vertex[0];
      const T2& v1 = vertex[1];
      const T2& v2 = vertex[2];
      const T2& v3 = vertex[3];
  
      const T2 dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<T2>(px, py, pz, v0, v1, v2, sradius);
  
      if (dot_product > 0.00f) {
        t_size -= 1;
        r_size += 1;
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 0], 
                    T[3 * t_maxsize * i + 3 * t_size  + 0]);
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 1], 
                    T[3 * t_maxsize * i + 3 * t_size  + 1]);
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 2], 
                    T[3 * t_maxsize * i + 3 * t_size  + 2]);
        t_index -= 1;
      }
  
    }                                         
    
    if (r_size > 0) {

      if (p_size > p_maxsize) {
        return;
      }

      P[4 * p_maxsize * i + 4 * p_size + 0] = bisector[0];
      P[4 * p_maxsize * i + 4 * p_size + 1] = bisector[1];
      P[4 * p_maxsize * i + 4 * p_size + 2] = bisector[2];
      P[4 * p_maxsize * i + 4 * p_size + 3] = bisector[3];
      dknn[k * i + neighbor] = p_size;
      p_size += 1;
  
      const size_t dr_offs = p_maxsize * i;
      for (size_t j = 0; j < p_maxsize; j++) {
        dR[dr_offs + j] = boundary::bstatus::undefined;
      }

      short int head = -1;
      boundary::bstatus bstat = boundary::compute(
        dR, dr_offs, p_maxsize, head,
        T, 3 * t_maxsize * i + t_size * 3, r_size
      );
      r_size = 0;
  
      if (bstat == boundary::bstatus::unreachable) {
        state.set_true(cc::error_infinite_boundary);
        state.set_true(cc::error_occurred);
        return;
      }
  
      auto first = head;
      while (true) {

        if (t_size > t_maxsize) {
          return;
        }
  
        const auto nvertex_0 = head;
        const auto nvertex_1 = dR[dr_offs + nvertex_0];
        head = nvertex_1;
  
        T[3 * t_maxsize * i + 3 * t_size + 0] = nvertex_0;
        T[3 * t_maxsize * i + 3 * t_size + 1] = nvertex_1;
        T[3 * t_maxsize * i + 3 * t_size + 2] = p_size - 1;
        
        t_size += 1;
  
        if (head == first) break;
      }

    } else {
      state.set_true(cc::error_nonvalid_neighbor);
    }
  
    if (state.get(cc::error_occurred)) {
      state.set_false(cc::error_occurred);
    }
    if (state.get(cc::error_nonvalid_neighbor)) {
      state.set_false(cc::error_nonvalid_neighbor);
    }
    if (sr::is_reached(px,py,pz,qx,qy,qz,sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
  
  for (size_t di = 0; di < k; di++) {
    bool flag = true;
    for (size_t ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[k * i + di]) flag = false;
    } if (flag) knn[k * i + di] = cc::k_undefined;
  }

  size_t dnn_counter = 0;
  for (size_t di = 0; di < k; di++) {
    if (knn[k * i + di] == cc::k_undefined) continue;
    knn[k * i + dnn_counter] = knn[k * i + di];
    dnn_counter += 1;
  }
  for (size_t di = dnn_counter; di < k; di++) {
    knn[k * i + di] = cc::k_undefined;
  }
  
}

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2, typename T3>
void dnni::compute(
  const T1 i,
  const T1 index,
  std::vector<cc::state>& states,
  T2* P,
  T3* T,
  T3* dR,
  std::vector<T1>& knn,
  std::vector<T1>& dknn,
  const std::vector<std::array<T2,3>>& xyzset,
  const size_t xyzsize,
  const std::vector<std::array<T2,3>>& refset,
  const size_t refsize,
  const struct args::cc& args
) {

  (void) xyzsize;
  (void) refsize;
  
  // TODO : make this swappable for future
  static const T2 p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const T3 t_init[] = { 
    2,5,0, 5,3,0, 1,5,2, 5,1,3, 
    4,2,0, 4,0,3, 2,4,1, 4,3,1 
  };
  
  const unsigned short int p_initsize = sizeof(p_init) / (sizeof(*p_init) * 4);
  const unsigned short int t_initsize = sizeof(t_init) / (sizeof(*t_init) * 3);

  const unsigned short int k = args.k;
  const unsigned short int p_maxsize = args.p_maxsize;
  const unsigned short int t_maxsize = args.t_maxsize;

  unsigned short int p_size = p_initsize;
  unsigned short int t_size = t_initsize; 
  unsigned short int r_size = 0; 

  cc::state& state = states[index];

  T2 vertex[4];
  T2 bisector[4];

  const T2 px = refset[index][0];
  const T2 py = refset[index][1];
  const T2 pz = refset[index][2];
 
  for (size_t j = 0; j < p_initsize; j++) {
    P[4 * p_maxsize * i + 4 * j + 0] = p_init[4 * j + 0];
    P[4 * p_maxsize * i + 4 * j + 1] = p_init[4 * j + 1];
    P[4 * p_maxsize * i + 4 * j + 2] = p_init[4 * j + 2];
    P[4 * p_maxsize * i + 4 * j + 3] = p_init[4 * j + 3];
  }

  for (size_t j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }

  for (size_t neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[k * i + neighbor];
    const T2 qx = xyzset[q][0];
    const T2 qy = xyzset[q][1];
    const T2 qz = xyzset[q][2];

    r_size = 0;
    T2 sradius = 0.00f;
  
    planes::bisect<T2>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const T3& t0 = T[3 * t_maxsize * i + 3 * t_index + 0];
      const T3& t1 = T[3 * t_maxsize * i + 3 * t_index + 1];
      const T3& t2 = T[3 * t_maxsize * i + 3 * t_index + 2];
  
      const T2& plane_00 = P[4 * p_maxsize * i + 4 * t0 + 0];
      const T2& plane_01 = P[4 * p_maxsize * i + 4 * t0 + 1];
      const T2& plane_02 = P[4 * p_maxsize * i + 4 * t0 + 2];
      const T2& plane_03 = P[4 * p_maxsize * i + 4 * t0 + 3];

      const T2& plane_10 = P[4 * p_maxsize * i + 4 * t1 + 0];
      const T2& plane_11 = P[4 * p_maxsize * i + 4 * t1 + 1];
      const T2& plane_12 = P[4 * p_maxsize * i + 4 * t1 + 2];
      const T2& plane_13 = P[4 * p_maxsize * i + 4 * t1 + 3];

      const T2& plane_20 = P[4 * p_maxsize * i + 4 * t2 + 0];
      const T2& plane_21 = P[4 * p_maxsize * i + 4 * t2 + 1];
      const T2& plane_22 = P[4 * p_maxsize * i + 4 * t2 + 2];
      const T2& plane_23 = P[4 * p_maxsize * i + 4 * t2 + 3];

      // TODO : implement exception handling
      planes::intersect<T2>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const T2& b0 = bisector[0];
      const T2& b1 = bisector[1];
      const T2& b2 = bisector[2];
      const T2& b3 = bisector[3];
  
      const T2& v0 = vertex[0];
      const T2& v1 = vertex[1];
      const T2& v2 = vertex[2];
      const T2& v3 = vertex[3];
  
      const T2 dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<T2>(px, py, pz, v0, v1, v2, sradius);
  
      if (dot_product > 0.00f) {
        t_size -= 1;
        r_size += 1;
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 0], 
                    T[3 * t_maxsize * i + 3 * t_size  + 0]);
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 1], 
                    T[3 * t_maxsize * i + 3 * t_size  + 1]);
        utils::swap(T[3 * t_maxsize * i + 3 * t_index + 2], 
                    T[3 * t_maxsize * i + 3 * t_size  + 2]);
        t_index -= 1;
      }
  
    }                                         
    
    if (r_size > 0) {

      if (p_size > p_maxsize) {
        return;
      }

      P[4 * p_maxsize * i + 4 * p_size + 0] = bisector[0];
      P[4 * p_maxsize * i + 4 * p_size + 1] = bisector[1];
      P[4 * p_maxsize * i + 4 * p_size + 2] = bisector[2];
      P[4 * p_maxsize * i + 4 * p_size + 3] = bisector[3];
      dknn[k * i + neighbor] = p_size;
      p_size += 1;
  
      const size_t dr_offs = p_maxsize * i;
      for (size_t j = 0; j < p_maxsize; j++) {
        dR[dr_offs + j] = boundary::bstatus::undefined;
      }

      short int head = -1;
      boundary::bstatus bstat = boundary::compute(
        dR, dr_offs, p_maxsize, head,
        T, 3 * t_maxsize * i + t_size * 3, r_size
      );
      r_size = 0;
  
      if (bstat == boundary::bstatus::unreachable) {
        state.set_true(cc::error_infinite_boundary);
        state.set_true(cc::error_occurred);
        return;
      }
  
      auto first = head;
      while (true) {

        if (t_size > t_maxsize) {
          return;
        }
  
        const auto nvertex_0 = head;
        const auto nvertex_1 = dR[dr_offs + nvertex_0];
        head = nvertex_1;
  
        T[3 * t_maxsize * i + 3 * t_size + 0] = nvertex_0;
        T[3 * t_maxsize * i + 3 * t_size + 1] = nvertex_1;
        T[3 * t_maxsize * i + 3 * t_size + 2] = p_size - 1;
        
        t_size += 1;
  
        if (head == first) break;
      }

    } else {
      state.set_true(cc::error_nonvalid_neighbor);
    }
  
    if (state.get(cc::error_occurred)) {
      state.set_false(cc::error_occurred);
    }
    if (state.get(cc::error_nonvalid_neighbor)) {
      state.set_false(cc::error_nonvalid_neighbor);
    }
    if (sr::is_reached(px,py,pz,qx,qy,qz,sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
  
  for (size_t di = 0; di < k; di++) {
    bool flag = true;
    for (size_t ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[k * i + di]) flag = false;
    } if (flag) knn[k * i + di] = cc::k_undefined;
  }

  size_t dnn_counter = 0;
  for (size_t di = 0; di < k; di++) {
    if (knn[k * i + di] == cc::k_undefined) continue;
    knn[k * i + dnn_counter] = knn[k * i + di];
    dnn_counter += 1;
  }
  for (size_t di = dnn_counter; di < k; di++) {
    knn[k * i + di] = cc::k_undefined;
  }
  
}

///////////////////////////////////////////////////////////////////////////////
/// Sycl Implementation
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template <typename T1, typename T2, typename T3>
void dnni::compute(
  const T1 i,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<T2>& P,
  const device_accessor_readwrite_t<T3>& T,
  const device_accessor_readwrite_t<T3>& dR,
  const device_accessor_readwrite_t<T1>& knn,
  const device_accessor_readwrite_t<T1>& dknn,
  const device_accessor_read_t<T2>& xyzset,
  const size_t xyzsize,
  const device_accessor_read_t<T2>& refset,
  const size_t refsize,
  const struct args::cc& args
) {
  
  // TODO : make this swappable for future
  static const T2 p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const T3 t_init[] = { 
    2,5,0, 5,3,0, 1,5,2, 5,1,3, 
    4,2,0, 4,0,3, 2,4,1, 4,3,1 
  };

  const unsigned short int p_initsize = sizeof(p_init) / (sizeof(*p_init) * 4);
  const unsigned short int t_initsize = sizeof(t_init) / (sizeof(*t_init) * 3);

  const unsigned short int k = args.k;
  const unsigned short int p_maxsize = args.p_maxsize;
  const unsigned short int t_maxsize = args.t_maxsize;

  unsigned short int p_size = p_initsize;
  unsigned short int t_size = t_initsize; 
  unsigned short int r_size = 0; 

  cc::state& state = states[i];

  T2 vertex[4];
  T2 bisector[4];

  const T2 px = refset[refsize * 0 + i];
  const T2 py = refset[refsize * 1 + i];
  const T2 pz = refset[refsize * 2 + i];
  
  for (size_t j = 0; j < p_initsize; j++) {
    P[4 * refsize * j + refsize * 0 + i] = p_init[4 * j + 0];
    P[4 * refsize * j + refsize * 1 + i] = p_init[4 * j + 1];
    P[4 * refsize * j + refsize * 2 + i] = p_init[4 * j + 2];
    P[4 * refsize * j + refsize * 3 + i] = p_init[4 * j + 3];
  }

  for (size_t j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }
  
  for (size_t neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[k * i + neighbor];
    const T2 qx = xyzset[xyzsize * 0 + q];
    const T2 qy = xyzset[xyzsize * 1 + q];
    const T2 qz = xyzset[xyzsize * 2 + q];
  
    r_size = 0;
    T2 sradius = 0.00f;
  
    planes::bisect<T2>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const T3& t0 = T[3 * t_maxsize * i + t_index * 3 + 0];
      const T3& t1 = T[3 * t_maxsize * i + t_index * 3 + 1];
      const T3& t2 = T[3 * t_maxsize * i + t_index * 3 + 2];
  
      const T2& plane_00 = P[4 * refsize * t0 + refsize * 0 + i];
      const T2& plane_01 = P[4 * refsize * t0 + refsize * 1 + i];
      const T2& plane_02 = P[4 * refsize * t0 + refsize * 2 + i];
      const T2& plane_03 = P[4 * refsize * t0 + refsize * 3 + i];

      const T2& plane_10 = P[4 * refsize * t1 + refsize * 0 + i];
      const T2& plane_11 = P[4 * refsize * t1 + refsize * 1 + i];
      const T2& plane_12 = P[4 * refsize * t1 + refsize * 2 + i];
      const T2& plane_13 = P[4 * refsize * t1 + refsize * 3 + i];

      const T2& plane_20 = P[4 * refsize * t2 + refsize * 0 + i];
      const T2& plane_21 = P[4 * refsize * t2 + refsize * 1 + i];
      const T2& plane_22 = P[4 * refsize * t2 + refsize * 2 + i];
      const T2& plane_23 = P[4 * refsize * t2 + refsize * 3 + i];
  
      // TODO : implement exception handling
      planes::intersect<T2>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const T2& b0 = bisector[0];
      const T2& b1 = bisector[1];
      const T2& b2 = bisector[2];
      const T2& b3 = bisector[3];
  
      const T2& v0 = vertex[0];
      const T2& v1 = vertex[1];
      const T2& v2 = vertex[2];
      const T2& v3 = vertex[3];
  
      const T2 dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<T2>(px, py, pz, v0, v1, v2, sradius);
  
      if (dot_product > 0.00f) {
        t_size -= 1;
        r_size += 1;
        utils::swap(T[3 * t_maxsize * i + t_index * 3 + 0], 
                    T[3 * t_maxsize * i + t_size  * 3 + 0]);
        utils::swap(T[3 * t_maxsize * i + t_index * 3 + 1], 
                    T[3 * t_maxsize * i + t_size  * 3 + 1]);
        utils::swap(T[3 * t_maxsize * i + t_index * 3 + 2], 
                    T[3 * t_maxsize * i + t_size  * 3 + 2]);
        t_index -= 1;
      }
  
    }                                         
    
    if (r_size > 0) {

      if (p_size > p_maxsize) {
        return;
      }

      P[4 * refsize * p_size + refsize * 0 + i] = bisector[0];
      P[4 * refsize * p_size + refsize * 1 + i] = bisector[1];
      P[4 * refsize * p_size + refsize * 2 + i] = bisector[2];
      P[4 * refsize * p_size + refsize * 3 + i] = bisector[3];
      dknn[k * i + neighbor] = p_size;
      p_size += 1;
  
      const size_t dr_offs = p_maxsize * i;
      for (size_t j = 0; j < p_maxsize; j++) {
        dR[dr_offs + j] = boundary::bstatus::undefined;
      }

      short int head = -1;
      boundary::bstatus bstat = boundary::compute(
        dR, dr_offs, p_maxsize, head,
        T, 3 * t_maxsize * i + t_size * 3, r_size
      );
      r_size = 0;
  
      if (bstat == boundary::bstatus::unreachable) {
        state.set_true(cc::error_infinite_boundary);
        state.set_true(cc::error_occurred);
        return;
      }
  
      auto first = head;
      while (true) {

        if (t_size > t_maxsize) {
          return;
        }
  
        const auto nvertex_0 = head;
        const auto nvertex_1 = dR[dr_offs + nvertex_0];
        head = nvertex_1;
  
        T[3 * t_maxsize * i + t_size * 3 + 0] = nvertex_0;
        T[3 * t_maxsize * i + t_size * 3 + 1] = nvertex_1;
        T[3 * t_maxsize * i + t_size * 3 + 2] = p_size - 1;
        
        t_size += 1;
  
        if (head == first) break;
      }

    } else {
      state.set_true(cc::error_nonvalid_neighbor);
    }
  
    if (state.get(cc::error_occurred)) {
      std::cerr << "oh no error occured\n";
      state.set_false(cc::error_occurred);
    }
    if (state.get(cc::error_nonvalid_neighbor)) {
      state.set_false(cc::error_nonvalid_neighbor);
    }
    if (sr::is_reached(px,py,pz,qx,qy,qz,sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
  
  for (size_t di = 0; di < k; di++) {
    bool flag = true;
    for (size_t ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[k * i + di]) flag = false;
    } if (flag) knn[k * i + di] = cc::k_undefined;
  }

  size_t dnn_counter = 0;
  for (size_t di = 0; di < k; di++) {
    if (knn[k * i + di] == cc::k_undefined) continue;
    knn[k * i + dnn_counter] = knn[k * i + di];
    dnn_counter += 1;
  }
  for (size_t di = dnn_counter; di < k; di++) {
    knn[k * i + di] = cc::k_undefined;
  }
  
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
