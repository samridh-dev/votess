template <typename Ti, typename Tf, typename Tu>
void cci::compute(
  const Ti i, const Ti index,
  std::vector<cc::state>& states,
  Tf* P,
  Tu* T,
  Tu* dR,
  std::vector<Ti>& knn,
  std::vector<Ti>& dknn,
  const std::vector<std::array<Tf,3>>& xyzset,
  const Ti xyzsize,
  const std::vector<std::array<Tf,3>>& refset,
  const Ti refsize,
  const struct args::cc& args
) {

  (void) xyzsize;
  (void) refsize;
  
  // TODO : make this swappable for future
  static const Tf p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const Tu t_init[] = { 
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

  Tf vertex[4];
  Tf bisector[4];

  const Tf px = refset[index][0];
  const Tf py = refset[index][1];
  const Tf pz = refset[index][2];
 
  for (Ti j = 0; j < p_initsize; j++) {
    P[4 * p_maxsize * i + 4 * j + 0] = p_init[4 * j + 0];
    P[4 * p_maxsize * i + 4 * j + 1] = p_init[4 * j + 1];
    P[4 * p_maxsize * i + 4 * j + 2] = p_init[4 * j + 2];
    P[4 * p_maxsize * i + 4 * j + 3] = p_init[4 * j + 3];
  }

  for (Ti j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }

  for (Ti neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[k * i + neighbor];
    const Tf qx = xyzset[q][0];
    const Tf qy = xyzset[q][1];
    const Tf qz = xyzset[q][2];

    r_size = 0;
    Tf sradius = 0.00f;
  
    planes::bisect<Tf>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const Tu& t0 = T[3 * t_maxsize * i + 3 * t_index + 0];
      const Tu& t1 = T[3 * t_maxsize * i + 3 * t_index + 1];
      const Tu& t2 = T[3 * t_maxsize * i + 3 * t_index + 2];
  
      const Tf& plane_00 = P[4 * p_maxsize * i + 4 * t0 + 0];
      const Tf& plane_01 = P[4 * p_maxsize * i + 4 * t0 + 1];
      const Tf& plane_02 = P[4 * p_maxsize * i + 4 * t0 + 2];
      const Tf& plane_03 = P[4 * p_maxsize * i + 4 * t0 + 3];

      const Tf& plane_10 = P[4 * p_maxsize * i + 4 * t1 + 0];
      const Tf& plane_11 = P[4 * p_maxsize * i + 4 * t1 + 1];
      const Tf& plane_12 = P[4 * p_maxsize * i + 4 * t1 + 2];
      const Tf& plane_13 = P[4 * p_maxsize * i + 4 * t1 + 3];

      const Tf& plane_20 = P[4 * p_maxsize * i + 4 * t2 + 0];
      const Tf& plane_21 = P[4 * p_maxsize * i + 4 * t2 + 1];
      const Tf& plane_22 = P[4 * p_maxsize * i + 4 * t2 + 2];
      const Tf& plane_23 = P[4 * p_maxsize * i + 4 * t2 + 3];

      // TODO : implement exception handling
      planes::intersect<Tf>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const Tf& b0 = bisector[0];
      const Tf& b1 = bisector[1];
      const Tf& b2 = bisector[2];
      const Tf& b3 = bisector[3];
  
      const Tf& v0 = vertex[0];
      const Tf& v1 = vertex[1];
      const Tf& v2 = vertex[2];
      const Tf& v3 = vertex[3];
  
      const Tf dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<Tf>(px, py, pz, v0, v1, v2, sradius);
  
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
        state.set_true(cc::error_p_overflow);
        return;
      }

      P[4 * p_maxsize * i + 4 * p_size + 0] = bisector[0];
      P[4 * p_maxsize * i + 4 * p_size + 1] = bisector[1];
      P[4 * p_maxsize * i + 4 * p_size + 2] = bisector[2];
      P[4 * p_maxsize * i + 4 * p_size + 3] = bisector[3];
      dknn[k * i + neighbor] = p_size;
      p_size += 1;
  
      const Ti dr_offs = p_maxsize * i;
      for (Ti j = 0; j < p_maxsize; j++) {
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
          state.set_true(cc::error_t_overflow);
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
    if (sr::is_reached(px, py, pz, qx, qy, qz, sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
  
  for (Ti di = 0; di < k; di++) {
    bool flag = true;
    for (Ti ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[k * i + di]) flag = false;
    } if (flag) knn[k * i + di] = cc::k_undefined;
  }

  Ti dnn_counter = 0;
  for (Ti di = 0; di < k; di++) {
    if (knn[k * i + di] == cc::k_undefined) continue;
    knn[k * i + dnn_counter] = knn[k * i + di];
    dnn_counter += 1;
  }
  for (Ti di = dnn_counter; di < k; di++) {
    knn[k * i + di] = cc::k_undefined;
  }
  
}

template <typename Ti, typename Tf, typename Tu>
void cci::compute(
  const Ti i, const Ti index,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<Tf>& P,
  const device_accessor_readwrite_t<Tu>& T,
  const device_accessor_readwrite_t<Tu>& dR,
  const device_accessor_readwrite_t<Ti>& knn,
  const device_accessor_readwrite_t<Ti>& dknn,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const struct args::cc& args
) {
  
  // TODO : make this swappable for future
  static const Tf p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const Tu t_init[] = { 
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

  Tf vertex[4];
  Tf bisector[4];

  const Tf px = refset[refsize * 0 + index];
  const Tf py = refset[refsize * 1 + index];
  const Tf pz = refset[refsize * 2 + index];
  
  for (Ti j = 0; j < p_initsize; j++) {
    P[4 * refsize * j + refsize * 0 + i] = p_init[4 * j + 0];
    P[4 * refsize * j + refsize * 1 + i] = p_init[4 * j + 1];
    P[4 * refsize * j + refsize * 2 + i] = p_init[4 * j + 2];
    P[4 * refsize * j + refsize * 3 + i] = p_init[4 * j + 3];
  }

  for (Ti j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }
  
  for (Ti neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[k * i + neighbor];
    const Tf qx = xyzset[xyzsize * 0 + q];
    const Tf qy = xyzset[xyzsize * 1 + q];
    const Tf qz = xyzset[xyzsize * 2 + q];
  
    r_size = 0;
    Tf sradius = 0.00f;
  
    planes::bisect<Tf>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const Tu& t0 = T[3 * t_maxsize * i + t_index * 3 + 0];
      const Tu& t1 = T[3 * t_maxsize * i + t_index * 3 + 1];
      const Tu& t2 = T[3 * t_maxsize * i + t_index * 3 + 2];
  
      const Tf& plane_00 = P[4 * refsize * t0 + refsize * 0 + i];
      const Tf& plane_01 = P[4 * refsize * t0 + refsize * 1 + i];
      const Tf& plane_02 = P[4 * refsize * t0 + refsize * 2 + i];
      const Tf& plane_03 = P[4 * refsize * t0 + refsize * 3 + i];

      const Tf& plane_10 = P[4 * refsize * t1 + refsize * 0 + i];
      const Tf& plane_11 = P[4 * refsize * t1 + refsize * 1 + i];
      const Tf& plane_12 = P[4 * refsize * t1 + refsize * 2 + i];
      const Tf& plane_13 = P[4 * refsize * t1 + refsize * 3 + i];

      const Tf& plane_20 = P[4 * refsize * t2 + refsize * 0 + i];
      const Tf& plane_21 = P[4 * refsize * t2 + refsize * 1 + i];
      const Tf& plane_22 = P[4 * refsize * t2 + refsize * 2 + i];
      const Tf& plane_23 = P[4 * refsize * t2 + refsize * 3 + i];
  
      // TODO : implement exception handling
      planes::intersect<Tf>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const Tf& b0 = bisector[0];
      const Tf& b1 = bisector[1];
      const Tf& b2 = bisector[2];
      const Tf& b3 = bisector[3];
  
      const Tf& v0 = vertex[0];
      const Tf& v1 = vertex[1];
      const Tf& v2 = vertex[2];
      const Tf& v3 = vertex[3];
  
      const Tf dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<Tf>(px, py, pz, v0, v1, v2, sradius);
  
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
        state.set_true(cc::error_p_overflow);
        return;
      }

      P[4 * refsize * p_size + refsize * 0 + i] = bisector[0];
      P[4 * refsize * p_size + refsize * 1 + i] = bisector[1];
      P[4 * refsize * p_size + refsize * 2 + i] = bisector[2];
      P[4 * refsize * p_size + refsize * 3 + i] = bisector[3];
      dknn[k * i + neighbor] = p_size;
      p_size += 1;
  
      const Ti dr_offs = p_maxsize * i;
      for (Ti j = 0; j < p_maxsize; j++) {
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
          state.set_true(cc::error_t_overflow);
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
      state.set_false(cc::error_occurred);
    }
    if (state.get(cc::error_nonvalid_neighbor)) {
      state.set_false(cc::error_nonvalid_neighbor);
    }
    if (sr::is_reached(px, py, pz, qx, qy, qz, sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
  
  for (Ti di = 0; di < k; di++) {
    bool flag = true;
    for (Ti ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[k * i + di]) flag = false;
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[k * i + di]) flag = false;
    } if (flag) knn[k * i + di] = cc::k_undefined;
  }

  Ti dnn_counter = 0;
  for (Ti di = 0; di < k; di++) {
    if (knn[k * i + di] == cc::k_undefined) continue;
    knn[k * i + dnn_counter] = knn[k * i + di];
    dnn_counter += 1;
  }
  for (Ti di = dnn_counter; di < k; di++) {
    knn[k * i + di] = cc::k_undefined;
  }
  
}

///////////////////////////////////////////////////////////////////////////////
/// New                                                                     ///
///////////////////////////////////////////////////////////////////////////////

template <typename Ti, typename Tf, typename Tu>
void cci::compute(
  const Ti i, const Ti index,
  const device_accessor_readwrite_t<cc::state>& states, 
  const device_accessor_readwrite_t<Tf>& P,
  const device_accessor_readwrite_t<Tu>& T,
  const device_accessor_readwrite_t<Tu>& dR,
  const device_accessor_readwrite_t<Ti>& knn, const Ti koffs,
  const device_accessor_readwrite_t<Ti>& dknn,
  const device_accessor_read_t<Tf>& xyzset,
  const Ti xyzsize,
  const device_accessor_read_t<Tf>& refset,
  const Ti refsize,
  const struct args::cc& args
) {
  
  // TODO : make this swappable for future
  static const Tf p_init[] = { 
    1,0,0,0, -1,0,0,1,
    0,1,0,0, 0,-1,0,1,
    0,0,1,0, 0,0,-1,1 
  };
  static const Tu t_init[] = { 
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

  Tf vertex[4];
  Tf bisector[4];

  const Tf px = refset[refsize * 0 + index];
  const Tf py = refset[refsize * 1 + index];
  const Tf pz = refset[refsize * 2 + index];
  
  for (Ti j = 0; j < p_initsize; j++) {
    P[4 * refsize * j + refsize * 0 + i] = p_init[4 * j + 0];
    P[4 * refsize * j + refsize * 1 + i] = p_init[4 * j + 1];
    P[4 * refsize * j + refsize * 2 + i] = p_init[4 * j + 2];
    P[4 * refsize * j + refsize * 3 + i] = p_init[4 * j + 3];
  }

  for (Ti j = 0; j < t_initsize; j++) {
    T[3 * t_maxsize * i + 3 * j + 0] = t_init[3 * j + 0];
    T[3 * t_maxsize * i + 3 * j + 1] = t_init[3 * j + 1];
    T[3 * t_maxsize * i + 3 * j + 2] = t_init[3 * j + 2];
  }
  
  for (Ti neighbor = 0; neighbor < k; neighbor++) {
  
    auto& q = knn[koffs * neighbor + i];
    const Tf qx = xyzset[xyzsize * 0 + q];
    const Tf qy = xyzset[xyzsize * 1 + q];
    const Tf qz = xyzset[xyzsize * 2 + q];
  
    r_size = 0;
    Tf sradius = 0.00f;
  
    planes::bisect<Tf>(
      bisector[0], bisector[1], bisector[2], bisector[3],
      qx, qy, qz, px, py, pz
    );
    
    for (short int t_index = 0; t_index < t_size; t_index++) {
  
      const Tu& t0 = T[3 * t_maxsize * i + t_index * 3 + 0];
      const Tu& t1 = T[3 * t_maxsize * i + t_index * 3 + 1];
      const Tu& t2 = T[3 * t_maxsize * i + t_index * 3 + 2];
  
      const Tf& plane_00 = P[4 * refsize * t0 + refsize * 0 + i];
      const Tf& plane_01 = P[4 * refsize * t0 + refsize * 1 + i];
      const Tf& plane_02 = P[4 * refsize * t0 + refsize * 2 + i];
      const Tf& plane_03 = P[4 * refsize * t0 + refsize * 3 + i];

      const Tf& plane_10 = P[4 * refsize * t1 + refsize * 0 + i];
      const Tf& plane_11 = P[4 * refsize * t1 + refsize * 1 + i];
      const Tf& plane_12 = P[4 * refsize * t1 + refsize * 2 + i];
      const Tf& plane_13 = P[4 * refsize * t1 + refsize * 3 + i];

      const Tf& plane_20 = P[4 * refsize * t2 + refsize * 0 + i];
      const Tf& plane_21 = P[4 * refsize * t2 + refsize * 1 + i];
      const Tf& plane_22 = P[4 * refsize * t2 + refsize * 2 + i];
      const Tf& plane_23 = P[4 * refsize * t2 + refsize * 3 + i];
  
      // TODO : implement exception handling
      planes::intersect<Tf>(
        vertex[0], vertex[1], vertex[2], vertex[3], 
        plane_00, plane_01, plane_02, plane_03,
        plane_10, plane_11, plane_12, plane_13,
        plane_20, plane_21, plane_22, plane_23
      ); 
      
      const Tf& b0 = bisector[0];
      const Tf& b1 = bisector[1];
      const Tf& b2 = bisector[2];
      const Tf& b3 = bisector[3];
  
      const Tf& v0 = vertex[0];
      const Tf& v1 = vertex[1];
      const Tf& v2 = vertex[2];
      const Tf& v3 = vertex[3];
  
      const Tf dot_product = planes::dot(v0, v1, v2, v3, b0, b1, b2, b3);
  
      sradius = sr::update<Tf>(px, py, pz, v0, v1, v2, sradius);
  
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
        state.set_true(cc::error_p_overflow);
        return;
      }

      P[4 * refsize * p_size + refsize * 0 + i] = bisector[0];
      P[4 * refsize * p_size + refsize * 1 + i] = bisector[1];
      P[4 * refsize * p_size + refsize * 2 + i] = bisector[2];
      P[4 * refsize * p_size + refsize * 3 + i] = bisector[3];
      dknn[koffs * neighbor + i] = p_size;
      p_size += 1;
  
      const Ti dr_offs = p_maxsize * i;
      for (Ti j = 0; j < p_maxsize; j++) {
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
          state.set_true(cc::error_t_overflow);
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
      state.set_false(cc::error_occurred);
    }
    if (state.get(cc::error_nonvalid_neighbor)) {
      state.set_false(cc::error_nonvalid_neighbor);
    }
    if (sr::is_reached(px, py, pz, qx, qy, qz, sradius)) {
      state.set_true(cc::security_radius_reached);
      break;
    }
  
  }
 
#if 0
  for (int di = 0; di < k; di++) {
    bool flag = true;
    for (Ti ti = 0; ti < t_size; ti++) {
      flag = !(T[3 * t_maxsize * i + 3 * ti + 0] == dknn[koffs * di + i] ||
               T[3 * t_maxsize * i + 3 * ti + 1] == dknn[koffs * di + i] ||
               T[3 * t_maxsize * i + 3 * ti + 2] == dknn[koffs * di + i]);
      if (!flag) break;
    } if (flag) knn[koffs * di + i] = cc::k_undefined;
  }
#else
  for (Ti di = 0; di < k; di++) {
    bool flag = true;
    for (Ti ti = 0; ti < t_size; ti++) {
      if (T[3 * t_maxsize * i + 3 * ti + 0] == dknn[koffs * di + i]) {
        flag = false;
      }
      if (T[3 * t_maxsize * i + 3 * ti + 1] == dknn[koffs * di + i]) {
        flag = false;
      }
      if (T[3 * t_maxsize * i + 3 * ti + 2] == dknn[koffs * di + i]) {
        flag = false;
      }
      if (!flag) break;
    } if (flag) knn[koffs * di + i] = cc::k_undefined;
  }
#endif

  Ti dnn_counter = 0;
  for (Ti di = 0; di < k; di++) {
    if (knn[koffs * di + i] == cc::k_undefined) continue;
    knn[koffs * dnn_counter + i] = knn[koffs * di + i];
    dnn_counter += 1;
  }
  for (Ti di = dnn_counter; di < k; di++) {
    knn[koffs * di + i] = cc::k_undefined;
  }
  
}
