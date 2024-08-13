import itertools
import numpy as np

def create_test(index, T, T_offs=0, dR_offs=0, dR_size=16, 
                ans=None, ulimit=10):
    """T has shape (N,3) and is an array of integers"""

    if ans is None: 
        ans = [0, 2, 1, 3]
    
    T_size = len(T) - T_offs
    
    T_subsection = T[T_offs:T_offs + T_size]

    max_permutations = 256

    print(f'TEST_CASE("[CPU] boundary tests: case {index}", "[boundary]") {{')
    print('')
    print(f'  const short int dR_size = {dR_size};')
    print(f'  std::vector<int> dR({dR_size * (1 + dR_offs)}, 0xff);')
    print(f'  std::vector<int> ans = {{{", ".join(map(str, ans))}}};')
    print('')
    print(f'  short int head = -1;')
    print(f'  short int dR_offs = {dR_offs * dR_size};')
    print(f'  short int T_offs = {T_offs * 3};')
    print(f'  short int T_size = {T_size};')
    print(f'  short int ulimit = {ulimit};')
    print('')
    
    counter = max_permutations
    for perm_index, perm in enumerate(itertools.permutations(T_subsection)):

        counter -= 1
        if counter <= 0: break
        
        T_flat = [item for sublist in (T[:T_offs].tolist() + list(perm) 
                                      + T[T_offs + T_size:].tolist()) 
                  for item in sublist]
        T_flat_str = ', '.join(map(str, T_flat))
        T_str = ', '.join(f"[{', '.join(map(str, row))}]" for row in perm)
        print(f'  SECTION("permutation {perm_index + 1}: {{{T_str}}}") {{')
        print(f'    std::vector<int> T = {{ {T_flat_str} }};')
        print('    boundary::compute(dR.data(), dR_offs, dR_size, head, ')
        print('                      T.data(), T_offs, T_size);')
        print('    std::vector<int> vec(dR.begin() + dR_offs, dR.end());')
        print('    boundary::test(vec, head, ans, ulimit);')
        print('  }')
        print('')

    print('}')

    print(f'TEST_CASE("[GPU] boundary tests: case {index}", "[boundary]") {{')
    print('')
    print(f'  const short int dR_size = {dR_size};')
    print(f'  std::vector<int> dR({dR_size * (1 + dR_offs)}, 0xff);')
    print(f'  std::vector<int> ans = {{{", ".join(map(str, ans))}}};')
    print('')
    print(f'  short int head = -1;')
    print(f'  short int dR_offs = {dR_offs * dR_size};')
    print(f'  short int T_offs = {T_offs * 3};')
    print(f'  short int T_size = {T_size};')
    print(f'  short int ulimit = {ulimit};')
    print('')
    
    counter = max_permutations
    for perm_index, perm in enumerate(itertools.permutations(T_subsection)):

        counter -= 1
        if counter <= 0: break
        
        T_flat = [item for sublist in (T[:T_offs].tolist() + list(perm) 
                                      + T[T_offs + T_size:].tolist()) 
                  for item in sublist]
        T_flat_str = ', '.join(map(str, T_flat))
        T_str = ', '.join(f"[{', '.join(map(str, row))}]" for row in perm)
        print(f'  SECTION("permutation {perm_index + 1}: {{{T_str}}}") {{')
        print(f'    std::vector<int> T = {{ {T_flat_str} }};')
        print('    boundary::compute_gpu(dR, dR_offs, dR_size, head,')
        print('                          T, T_offs, T_size);')
        print('    std::vector<int> vec(dR.begin() + dR_offs, dR.end());')
        print('    boundary::test(vec, head, ans, ulimit);')
        print('  }')
        print('')

    print('}')

def create_prefix():
    print('#include <catch2/catch_test_macros.hpp> ')
    print('#include <catch2/generators/catch_generators.hpp> ')
    print('#include <libsycl.hpp>')
    print('#include <boundary.hpp>')
    print('')
    print('#include <algorithm>')
    print('#include <vector>')
    print('#include <string>')
    print('#include <sstream>')
    print('')
    print('namespace boundary {')
    print('')
    print('template <typename T>')
    print('static void compute_gpu(')
    print('  std::vector<T>& cycle,')
    print('  const size_t dr_offs,')
    print('  const size_t dr_size,')
    print('  short int& head,')
    print('  std::vector<T>& R,')
    print('  const size_t r_offs,')
    print('  const size_t r_size) {')
    print('  ')
    print('  sycl::queue q;')
    print('  sycl::buffer<T>') 
    print('  bcycle(cycle.data(), sycl::range<1>(cycle.size()));')
    print('  sycl::buffer<T> bR(R.data(), sycl::range<1>(R.size()));')
    print('  sycl::buffer<short int> bhead(&head, sycl::range<1>(1));')
    print('')
    print('  q.submit([&](sycl::handler& h) {')
    print('    auto cycle = bcycle.template')
    print('    get_access<sycl::access::mode::read_write>(h);')
    print('    auto R = bR.template') 
    print('    get_access<sycl::access::mode::read_write>(h);')
    print('    auto ahead = bhead.template') 
    print('    get_access<sycl::access::mode::read_write>(h);')
    print('    h.single_task([=]() {')
    print('      boundary::compute(cycle, dr_offs, dr_size,')
    print('                        ahead[0], R, r_offs, r_size);')
    print('    });')
    print('  }).wait();')
    print('')
    print('  auto hcycle = bcycle.get_host_access();')
    print('  auto hR = bR.get_host_access();')
    print('  auto hhead = bhead.get_host_access();')
    print('')
    print('  std::copy(hcycle.get_pointer(), ')
    print('            hcycle.get_pointer() + cycle.size(),')
    print('            cycle.begin());')
    print('  std::copy(hR.get_pointer(), hR.get_pointer() + R.size(),')
    print('            R.begin());')
    print('  head = hhead[0];')
    print('')
    print('}')
    print('')
    print('} // namespace boundary')
    print('')
    print('static std::string ')
    print('vector_to_string(const std::vector<int>& vec) {')
    print('  std::stringstream ss;')
    print('  ss << "{";')
    print('  for (size_t i = 0; i < vec.size(); ++i) {')
    print('    ss << vec[i];')
    print('    if (i < vec.size() - 1) ss << ", ";')
    print('  }')
    print('  ss << "}";')
    print('  return ss.str();')
    print('}')
    print('')
    print('namespace boundary {')
    print('')
    print('static void test(')
    print('  const std::vector<int>& cycle,')
    print('  int head,')
    print('  const std::vector<int>& ans,')
    print('  const int ulimit')
    print(') {')
    print('')
    print('  const int first = head;')
    print('  int counter = 0;')
    print('')
    print('  INFO("Cycle: " << vector_to_string(cycle));')
    print('  INFO("Answer: " << vector_to_string(ans));')
    print('  REQUIRE(head < static_cast<int>(cycle.size()));')
    print('')
    print('  do {')
    print('    ')
    print('    REQUIRE(head < static_cast<int>(cycle.size()));')
    print('    const int vertex_0 = cycle[head];')
    print('    const int vertex_1 = cycle[vertex_0];')
    print('')
    print('    auto it_0 = std::find(ans.begin(), ans.end(), vertex_0);')
    print('    auto it_1 = std::find(ans.begin(), ans.end(), vertex_1);')
    print('')
    print('    if(it_0 == ans.end() || it_1 == ans.end()) {')
    print('      INFO("Missing vertex : " << vertex_0 << " or " << vertex_1);')
    print('      REQUIRE_FALSE("Cycle contains undesirable vertices" );')
    print('      break;')
    print('    }')
    print('')
    print('    const int ans_index_0 = std::distance(ans.begin(), it_0);')
    print('    const int ans_index_1 = std::distance(ans.begin(), it_1);')
    print('')
    print('    const int ans_size = ans.size();')
    print('    const bool b = ((ans_index_0 + 1) % ans_size == ans_index_1)||')
    print('                   ((ans_index_1 + 1) % ans_size == ans_index_0);')
    print('    REQUIRE(b);')
    print('')
    print('    head = cycle[head];')
    print('    REQUIRE(head < static_cast<int>(cycle.size()));')
    print('')
    print('    if (counter++ > ulimit) break;')
    print('')
    print('  } while (cycle[head] != first);')
    print('')
    print('}')
    print('')
    print('} // namespace boundary')

def main():
    index = 1

    create_prefix()

    create_test(index, np.array([[2,5,0], [5,3,0], [1,5,2], [5,1,3]]))
    index += 1

    create_test(index, np.array([[4,2,0], [4,0,3], [2,4,1], [4,3,1]]))
    index += 1

    create_test(index, np.array([[4,2,0], [4,0,3], [2,4,1], [4,3,1]]))
    index += 1

    create_test(index, np.array([[2,5,0], [5,3,0], [1,5,2]]),
                ans=[2,1,5,3,0])
    index += 1

    create_test(index, np.array([[4,2,0], [4,0,3], [2,4,1], [4,3,1]]),
                dR_offs = 23)
    index += 1

    create_test(index, np.array([[-1,-1,-1], [-1,-1,-1], [-1,-1,-1], 
                                [4,2,0], [4,0,3], [2,4,1], [4,3,1]]),
                dR_offs = 23, T_offs = 3)
    index += 1

    create_test(index, np.array([[-1,-1,-1], [-1,-1,-1], [-1,-1,-1],
                                 [-1,-1,-1], [-1,-1,-1],
                                 [1,5,2], [5,1,3], [4,2,0], [4,0,3], 
                                 [2,4,1], [4,3,1], [5,3,0]]),
                ans = [2,5,0],
                dR_offs = 11, T_offs = 5)
    index += 1

    create_test(index, np.array([[4,6,7], [0,4,7], [4,3,1], [6,1,8], 
                                 [1,3,8], [4,0,3], [4,1,6]]),
                ans = [0,3,8,6,7])
    index += 1

    create_test(index, np.array([[2,4,6], [1,2,6], [2,1,7], [1,6,7], 
                                 [6,0,7], [0,5,7], [5,2,7], [2,5,0]]),
                ans = [0,2,4,6])
    index += 1

if __name__ == '__main__':
    main()
