///////////////////////////////////////////////////////////////////////////////
/// Boundary Status                                                          //
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

inline 
boundary::vstatus::vstatus(void) : byte(0) {}

/* ------------------------------------------------------------------------- */

inline uint8_t 
boundary::vstatus::get(int b) {
  return (byte >> b) & 1;
}

/* ------------------------------------------------------------------------- */

inline void 
boundary::vstatus::set_true(int b) {
  byte |= 1 << b;
}

/* ------------------------------------------------------------------------- */

#define BOUNDARY_USE_STATIC_LIST 0

#if BOUNDARY_USE_STATIC_LIST
inline uint8_t 
boundary::vstatus::get_nshared_edges(void) const {
  static const uint8_t nbits[8] = {0,1,1,2,1,2,2,3}; 
  return nbits[byte];
}
#else
inline uint8_t 
boundary::vstatus::get_nshared_edges(void) const {
  switch (byte) {
    case 0: return 0;
    case 1: return 1;
    case 2: return 1;
    case 3: return 2;
    case 4: return 1;
    case 5: return 2;
    case 6: return 2;
    case 7: return 3;
    default: return 0;
  }
}
#endif

/* ------------------------------------------------------------------------- */

#if BOUNDARY_USE_STATIC_LIST
inline uint8_t 
boundary::vstatus::get_shared_position(void) const {
  static const uint8_t pbits[] = {
     255, // 0 0 0 0
        
     0,   // 0 0 0 1
     1,   // 0 0 1 0
     2,   // 0 0 1 1
        
     2,   // 0 1 0 0
     1,   // 0 1 0 1
     0,   // 0 1 1 0
        
     255, // 0 1 1 1
  };
  return pbits[byte];
}
#else
inline uint8_t 
boundary::vstatus::get_shared_position(void) const {
  switch (byte) {
    case 0: return 255; // 0 0 0 0
    case 1: return 0;   // 0 0 0 1
    case 2: return 1;   // 0 0 1 0
    case 3: return 2;   // 0 0 1 1
    case 4: return 2;   // 0 1 0 0
    case 5: return 1;   // 0 1 0 1
    case 6: return 0;   // 0 1 1 0
    case 7: return 255; // 0 1 1 1
    default: return 255; // if 'byte' > 7
  }
}
#endif

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// CPU Implementation                                                       //
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

#include <utils.hpp>
template<typename Ti, typename Tu>
inline boundary::bstatus boundary::compute(
  Tu* cycle, 
  const Ti dr_offs,
  const Ti dr_size,
  short int& head,
  Tu* R,
  const Ti r_offs,
  const Ti r_size
) {

  const uint8_t triangle_size = 3;
  for (uint8_t tri = 0; tri < triangle_size; tri++) {
    const Tu& triangle_edge_0 = R[r_offs + (tri + 0) % triangle_size];
    const Tu& triangle_edge_1 = R[r_offs + (tri + 1) % triangle_size];
    head = triangle_edge_0;
    cycle[dr_offs + head] = triangle_edge_1;
  }
  
  Ti swap_j = r_size;
  for (Ti j = 1; j < r_size; j++) {
  
    struct vstatus stat;
  
    const Tu first = head;
    for (uint8_t k = 0; k < triangle_size; k++) {
      
      const Tu& triangle_edge_0 = R[r_offs + j * 3 + (k + 0) % triangle_size];
      const Tu& triangle_edge_1 = R[r_offs + j * 3 + (k + 1) % triangle_size];
  
      uint8_t edge = cycle[dr_offs + first];
  
      Ti counter = 0;
      while (true) {
        head = edge;
  
        const Tu cycle_edge_0 = head;
        const Tu cycle_edge_1 = cycle[dr_offs + head];
  
        const bool cond_01 = cycle_edge_0 == triangle_edge_1;
        const bool cond_10 = cycle_edge_1 == triangle_edge_0;
        if ((cond_01 && cond_10)) {
          stat.set_true(k);
        }
        if (edge == first) break;
        edge = cycle[dr_offs + edge];
  
        if(counter++ > r_size) return bstatus::unreachable;
      }
    }
  
    if (j == swap_j) {
      return boundary::bstatus::unreachable;
    }
  
    const uint8_t nshared_edges = stat.get_nshared_edges();
    
    if (nshared_edges == 0) {
  
      swap_j -= 1;
      utils::swap(R[r_offs + j * 3 + 0], R[r_offs + swap_j * 3 + 0]);
      utils::swap(R[r_offs + j * 3 + 1], R[r_offs + swap_j * 3 + 1]);
      utils::swap(R[r_offs + j * 3 + 2], R[r_offs + swap_j * 3 + 2]);
      j -= 1;
      continue;
  
    } 

    const uint8_t k = stat.get_shared_position();
    const Tu& triangle_edge_0 = R[r_offs + j * 3 + (k + 0) % triangle_size];
    const Tu& triangle_edge_1 = R[r_offs + j * 3 + (k + 1) % triangle_size];
    const Tu& triangle_edge_2 = R[r_offs + j * 3 + (k + 2) % triangle_size];
    const Tu& triangle_edge_3 = R[r_offs + j * 3 + (k + 3) % triangle_size];
  
    const Tu cycle_old_0  = cycle[dr_offs + triangle_edge_0];
    const Tu cycle_old_1  = cycle[dr_offs + triangle_edge_1];
    const Tu cycle_old_2  = cycle[dr_offs + triangle_edge_2];
    const Tu cycle_old_00 = cycle[dr_offs + cycle_old_0];

    if (nshared_edges == 1) {
  
      cycle[dr_offs + triangle_edge_1] = triangle_edge_2;
      cycle[dr_offs + triangle_edge_2] = triangle_edge_3;
      head = triangle_edge_2; 
  
    } else if (nshared_edges == 2) {
      
      head = cycle[dr_offs + triangle_edge_0];
      cycle[dr_offs + head] = boundary::bstatus::undefined;
                                                            
      cycle[dr_offs + triangle_edge_0] = triangle_edge_1;
      head = triangle_edge_0;
  
    } else if (nshared_edges == 3) {
    } else {
      return boundary::bstatus::unreachable;
    }
  
    bool flag = false;
    for (uint8_t edge_i = 0; edge_i < dr_size; edge_i++) {
      if (cycle[dr_offs + edge_i] == boundary::bstatus::undefined) continue;

      for (uint8_t edge_j = edge_i + 1; edge_j < dr_size; edge_j++) {
        if (cycle[dr_offs + edge_j] == boundary::bstatus::undefined) continue;

        const bool cond = cycle[dr_offs + edge_i] == cycle[dr_offs + edge_j];
        if (!cond) continue;

        cycle[dr_offs + triangle_edge_0] = cycle_old_0;
        cycle[dr_offs + triangle_edge_1] = cycle_old_1;
        cycle[dr_offs + triangle_edge_2] = cycle_old_2;
        cycle[dr_offs + cycle_old_0] = cycle_old_00;

        swap_j -= 1;
        utils::swap(R[r_offs + j * 3 + 0], R[r_offs + swap_j * 3 + 0]);
        utils::swap(R[r_offs + j * 3 + 1], R[r_offs + swap_j * 3 + 1]);
        utils::swap(R[r_offs + j * 3 + 2], R[r_offs + swap_j * 3 + 2]);
        j -= 1;

        flag = true;
        break;
      }
      if (flag) break;
    }
    if (!flag) swap_j = r_size;
  }

  return bstatus::success;
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// SYCL Implementation                                                      //
///////////////////////////////////////////////////////////////////////////////

/* ------------------------------------------------------------------------- */

template<typename Ti, typename Tu>
inline boundary::bstatus boundary::compute(
  const sycl::local_accessor<Tu, 1>& cycle,
  const Ti dr_offs,
  const Ti dr_size,
  short int& head,
  const device_accessor_readwrite_t<Tu>& R, 
  const Ti  r_offs,
  const Ti  r_size
) {

  const uint8_t triangle_size = 3;
  for (uint8_t tri = 0; tri < triangle_size; tri++) {
    const Tu& triangle_edge_0 = R[r_offs + (tri + 0) % triangle_size];
    const Tu& triangle_edge_1 = R[r_offs + (tri + 1) % triangle_size];
    head = triangle_edge_0;
    cycle[dr_offs + head] = triangle_edge_1;
  }
  
  Tu swap_j = r_size;
  for (Ti j = 1; j < r_size; j++) {
  
    struct vstatus stat;
  
    const Tu first = head;
    for (uint8_t k = 0; k < triangle_size; k++) {
      
      const Tu& triangle_edge_0 = R[r_offs + j * 3 + (k + 0) % triangle_size];
      const Tu& triangle_edge_1 = R[r_offs + j * 3 + (k + 1) % triangle_size];
  
      uint8_t edge = cycle[dr_offs + first];
  
      Tu counter = 0;
      while (true) {
        head = edge;
  
        const Tu cycle_edge_0 = head;
        const Tu cycle_edge_1 = cycle[dr_offs + head];
  
        const bool cond_01 = cycle_edge_0 == triangle_edge_1;
        const bool cond_10 = cycle_edge_1 == triangle_edge_0;
        if ((cond_01 && cond_10)) {
          stat.set_true(k);
        }
        if (edge == first) break;
        edge = cycle[dr_offs + edge];
  
        if(counter++ > r_size) return bstatus::unreachable;
      }
    }
  
    if (j == swap_j) {
      return boundary::bstatus::unreachable;
    }
  
    const uint8_t nshared_edges = stat.get_nshared_edges();
    if (nshared_edges == 0) {
  
      swap_j -= 1;
      utils::swap(R[r_offs + j * 3 + 0], R[r_offs + swap_j * 3 + 0]);
      utils::swap(R[r_offs + j * 3 + 1], R[r_offs + swap_j * 3 + 1]);
      utils::swap(R[r_offs + j * 3 + 2], R[r_offs + swap_j * 3 + 2]);
      j -= 1;
      continue;
  
    } 
  
    const uint8_t k = stat.get_shared_position();
    const Tu& triangle_edge_0 = R[r_offs + j * 3 + (k + 0) % triangle_size];
    const Tu& triangle_edge_1 = R[r_offs + j * 3 + (k + 1) % triangle_size];
    const Tu& triangle_edge_2 = R[r_offs + j * 3 + (k + 2) % triangle_size];
    const Tu& triangle_edge_3 = R[r_offs + j * 3 + (k + 3) % triangle_size];
  
    const Tu cycle_old_0  = cycle[dr_offs + triangle_edge_0];
    const Tu cycle_old_1  = cycle[dr_offs + triangle_edge_1];
    const Tu cycle_old_2  = cycle[dr_offs + triangle_edge_2];
    const Tu cycle_old_00 = cycle[dr_offs + cycle_old_0];
  
    if (nshared_edges == 1) {
  
      cycle[dr_offs + triangle_edge_1] = triangle_edge_2;
      cycle[dr_offs + triangle_edge_2] = triangle_edge_3;
      head = triangle_edge_2; 
  
    } else if (nshared_edges == 2) {
      
      head = cycle[dr_offs + triangle_edge_0];
      cycle[dr_offs + head] = boundary::bstatus::undefined;
                                                            
      cycle[dr_offs + triangle_edge_0] = triangle_edge_1;
      head = triangle_edge_0;
  
    } else if (nshared_edges == 3) {
    } else {
      return boundary::bstatus::unreachable;
    }
  
  
    bool flag = false;
    for (uint8_t edge_i = 0; edge_i < dr_size; edge_i++) {
      if (cycle[dr_offs + edge_i] == boundary::bstatus::undefined) continue;


      for (uint8_t edge_j = edge_i + 1; edge_j < dr_size; edge_j++) {
        if (cycle[dr_offs + edge_j] == boundary::bstatus::undefined) continue;

        const bool cond = cycle[dr_offs + edge_i] == cycle[dr_offs + edge_j];
        if (!cond) continue;

        cycle[dr_offs + triangle_edge_0] = cycle_old_0;
        cycle[dr_offs + triangle_edge_1] = cycle_old_1;
        cycle[dr_offs + triangle_edge_2] = cycle_old_2;
        cycle[dr_offs + cycle_old_0] = cycle_old_00;

        swap_j -= 1;
        utils::swap(R[r_offs + j * 3 + 0], R[r_offs + swap_j * 3 + 0]);
        utils::swap(R[r_offs + j * 3 + 1], R[r_offs + swap_j * 3 + 1]);
        utils::swap(R[r_offs + j * 3 + 2], R[r_offs + swap_j * 3 + 2]);
        j -= 1;

        flag = true;
        break;
      }
      if (flag) break;
    }
    if (!flag) swap_j = r_size;
  }
  
  return bstatus::success;
}

/* ------------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
/// End
///////////////////////////////////////////////////////////////////////////////
