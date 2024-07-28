import numpy as np

def create_prefix():
    print('#include <catch2/catch_test_macros.hpp>')
    print('#include <catch2/matchers/catch_matchers_floating_point.hpp>')
    print('')
    print('#include <votess.hpp>')
    print('')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('/// Forward Declarations                                                    ///')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('')
    print('template <typename T>')
    print('static void test_votess(')
    print('  std::vector<std::array<T,3>>& xyzset,')
    print('  const int k,')
    print('  const int gr')
    print(');')
    print('')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('/// Test Cases                                                              ///')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('')

def create_suffix():
    print('///////////////////////////////////////////////////////////////////////////////')
    print('/// Internal Functions                                                      ///')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('')
    print('#include <iostream>')
    print('class __internal__suppress_stdout {')
    print('  public:')
    print('    __internal__suppress_stdout() : buf(std::cout.rdbuf()) {')
    print('      std::cout.rdbuf(__tmp__buf.rdbuf());')
    print('    }')
    print('    ~__internal__suppress_stdout() {')
    print('      std::cout.rdbuf(buf);')
    print('    }')
    print('  private:')
    print('    std::streambuf* buf;')
    print('    std::stringstream __tmp__buf;')
    print('};')
    print('')
    print('#include <voro++.hh>')
    print('template <typename T>')
    print('static std::pair<std::vector<std::array<T, 3>>, std::vector<std::vector<int>>>')
    print('run_voro(')
    print('  const std::vector<std::array<T,3>>& xyzset,')
    print('  const struct votess::vtargs vtargs')
    print(') {')
    print('  std::vector<std::array<T, 3>> coords;')
    print('  std::vector<std::vector<int>> neighbor_list;')
    print('  const double tolerance = 1e-8;')
    print('')
    print('  using namespace voro;')
    print('  container con(')
    print('    0, 1, 0, 1, 0, 1,')
    print('    vtargs.xyzset.grid_resolution,')
    print('    vtargs.xyzset.grid_resolution,')
    print('    vtargs.xyzset.grid_resolution,')
    print('    false, false, false,')
    print('    xyzset.size()')
    print('  );')
    print('  ')
    print('  for (size_t i = 0; i < xyzset.size(); i++) {')
    print('    con.put(i,xyzset[i][0],xyzset[i][1], xyzset[i][2]);')
    print('  }')
    print('')
    print('  c_loop_all cl(con);')
    print('  voronoicell_neighbor c;')
    print('  if (cl.start()) do if (con.compute_cell(c, cl)) {')
    print('    std::vector<int> neighbors;')
    print('    std::vector<int> filtered_neighbors;')
    print('    std::vector<double> face_areas;')
    print('    double x, y, z;')
    print('')
    print('    cl.pos(x, y, z);')
    print('    c.neighbors(neighbors);')
    print('    c.face_areas(face_areas);')
    print('')
    print('    for (size_t i = 0; i < face_areas.size(); i++) {')
    print('      if (face_areas[i] >= tolerance) {')
    print('        filtered_neighbors.push_back(neighbors[i]);')
    print('      }')
    print('    }')
    print('')
    print('    neighbor_list.push_back(filtered_neighbors);')
    print('    coords.push_back({')
    print('      static_cast<T>(x),')
    print('      static_cast<T>(y),')
    print('      static_cast<T>(z)')
    print('    });')
    print('')
    print('  } while(cl.inc());')
    print('')
    print('  return {coords, neighbor_list};')
    print('}')
    print('')
    print('template <typename T>')
    print('static void run_test(')
    print('  std::vector<std::array<T,3>>& xyzset,')
    print('  struct votess::vtargs vtargs,')
    print('  const enum votess::device device')
    print(') {')
    print('    __internal__suppress_stdout s; // to preventstdout ')
    print('')
    print('    (void)xyzset::sort<int,T>(xyzset, vtargs.xyzset);')
    print('')
    print('    auto [vcoord, vneighbor] = run_voro<T>(xyzset, vtargs);')
    print('    auto dnn = votess::tesellate<int, T>(xyzset, vtargs, device);')
    print('')
    print('    std::vector<int> test_dnn(0);')
    print('    std::vector<int> test_vneighbor(0);')
    print('')
    print('    for (size_t i = 0; i < xyzset.size(); i++) {')
    print('        auto it = std::find_if(')
    print('            vcoord.begin(),')
    print('            vcoord.end(),')
    print('            [&](const std::array<T, 3>& elem) {')
    print('                return (elem[0] == xyzset[i][0]) &&')
    print('                       (elem[1] == xyzset[i][1]) &&')
    print('                       (elem[2] == xyzset[i][2]);')
    print('            });')
    print('')
    print('        if (it == vcoord.end()) {')
    print('            CAPTURE(i);')
    print('            WARN("Matching coordinate not found in vcoord");')
    print('            continue;')
    print('        }')
    print('')
    print('        const size_t index = std::distance(vcoord.begin(), it);')
    print('        CAPTURE(index);')
    print('')
    print('        test_dnn.clear();')
    print('        test_vneighbor.clear();')
    print('        for (const auto& j : vneighbor[index]) {')
    print('            if (j < 0) continue;')
    print('            test_vneighbor.push_back(j);')
    print('        }')
    print('        for (size_t j = 0; j < dnn[i].size(); j++) {')
    print('            test_dnn.push_back(dnn[i][j]);')
    print('        }')
    print('')
    print('        std::sort(test_vneighbor.begin(), test_vneighbor.end());')
    print('        std::sort(test_dnn.begin(), test_dnn.end());')
    print('')
    print('        CAPTURE(test_vneighbor, test_dnn);')
    print('')
    print('        bool cond = std::includes(')
    print('          test_dnn.begin(),')
    print('          test_dnn.end(),')
    print('          test_vneighbor.begin(),')
    print('          test_vneighbor.end()')
    print('        );')
    print('        CAPTURE(cond);')
    print('        REQUIRE(cond);')
    print('    }')
    print('}')
    print('')
    print('template <typename T>')
    print('static void test_votess(')
    print('  std::vector<std::array<T,3>>& xyzset,')
    print('  const int k,')
    print('  const int gr')
    print(') {')
    print('  SECTION("(CPU) case : grid_resolution = " + std::to_string(gr)) {')
    print('    struct votess::vtargs vtargs(k,gr);')
    print('    run_test(xyzset, vtargs, votess::device::cpu);')
    print('  }')
    print('  SECTION("(GPU) case : grid_resolution = " + std::to_string(gr)) {')
    print('    struct votess::vtargs vtargs(k,gr);')
    print('    run_test(xyzset, vtargs, votess::device::gpu);')
    print('  }')
    print('}')
    print('')
    print('///////////////////////////////////////////////////////////////////////////////')
    print('/// End                                                                     ///')
    print('///////////////////////////////////////////////////////////////////////////////', end='')

def create_test_case(tag, i, points, gr_arr):


    print(f'TEST_CASE("votess regression {i}: {tag}", "[votess]") {{')
    print(f'\n  const int k = {len(points) - 1};')
    print('  std::vector<std::array<float, 3>> xyzset = {')
    
    for point in points:
        print(f'    {{{point[0]}f, {point[1]}f, {point[2]}f}},')
    
    print('  };')
    print()
    
    gr_str = "{" + ",".join(map(str, gr_arr)) + "}"
    print(f'  for (const auto& gr : {gr_str} ) {{')
    print(f'    test_votess(xyzset, k, gr);')
    print('  }')
    print()
    print('}')
    print()

###############################################################################
### Test Cases                                                              ###
###############################################################################

def create_standard_xyzset():
    return np.array([
        [0.605223, 0.108484, 0.090937],
        [0.500792, 0.499641, 0.464576],
        [0.437936, 0.786332, 0.160392],
        [0.663354, 0.170894, 0.810284],
        [0.614869, 0.096867, 0.204147],
        [0.556911, 0.895342, 0.802266],
        [0.305748, 0.124146, 0.516249],
        [0.406888, 0.157835, 0.919622],
        [0.094412, 0.861991, 0.798644],
        [0.511958, 0.560537, 0.345479]
    ])

def create_random_xyzset(n_points=128):
    return np.random.rand(n_points, 3)

def create_clustered_xyzset(center, n_points=100, scale=0.1):
    return np.random.normal(loc=center, scale=scale, size=(n_points, 3))

def create_lattice_xyzset(grid_size=3):
    return np.array([[x, y, z] for x in np.linspace(0.1, 0.9, grid_size)
                                for y in np.linspace(0.1, 0.9, grid_size)
                                for z in np.linspace(0.1, 0.9, grid_size)])

def create_fibonacci_sphere_xyzset(n_points=128):
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = 2 * np.pi * indices / golden_ratio
    theta = np.arccos(1 - 2*indices/n_points)
    
    # Spherical to Cartesian coordinates conversion
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    
    # Normalizing coordinates to fit within (0,1) exclusive
    min_val = min(x.min(), y.min(), z.min())
    max_val = max(x.max(), y.max(), z.max())
    x = (x - min_val) / (max_val - min_val)
    y = (y - min_val) / (max_val - min_val)
    z = (z - min_val) / (max_val - min_val)
    
    # Ensuring values are strictly within (0,1) exclusive
    epsilon = 1e-6
    x = np.clip(x, epsilon, 1 - epsilon)
    y = np.clip(y, epsilon, 1 - epsilon)
    z = np.clip(z, epsilon, 1 - epsilon)
    
    return np.column_stack((x, y, z))

def create_two_points_epsilon_xyzset(epsilon=1e-8):
    return np.array([
        [0.5, 0.5, 0.5],
        [0.5 + epsilon, 0.5, 0.5]
    ])

def create_degenerate_xyzset(N=128):
    point = np.array([[0.5, 0.5, 0.5]])
    return np.tile(point, (N, 1))

def create_sparse_xyzset(num_points=100, sparsity_level=0.95):
    data = np.random.rand(num_points, 3)
    mask = np.random.rand(num_points, 3) > sparsity_level
    sparse_data = data * mask
    return np.clip(sparse_data, 1e-6, 1 - 1e-6)

def create_noisy_xyzset(base_set, noise_level=0.1):
    noise = np.random.randn(*base_set.shape) * noise_level
    noisy_data = base_set + noise
    return np.clip(noisy_data, 1e-6, 1 - 1e-6)

def create_outliers_xyzset(base_set, outlier_factor=10):
    num_outliers = 5
    outliers = np.random.rand(num_outliers, 3) * outlier_factor
    outliers = np.clip(outliers, 1e-6, 1 - 1e-6)
    return np.vstack((base_set, outliers))

def create_imbalanced_class_xyzset(majority_class_size=1000,
                                   minority_class_size=10):
    majority_class = np.random.rand(majority_class_size, 3)
    minority_class = (np.random.rand(minority_class_size, 3) * 0.1) + 0.9
    majority_class = np.clip(majority_class, 1e-10, 1 - 1e-10)
    minority_class = np.clip(minority_class, 1e-10, 1 - 1e-10)
    return np.vstack((majority_class, minority_class))

def create_collinear_xyzset(x_range=(0.01, 0.99),
                            y_value=0.5, z_value=0.5, n_points=128):

    return np.array([[x, y_value, z_value] 
                     for x in np.linspace(x_range[0], x_range[1], n_points)])

def create_concentric_xyzset(center=np.array([0.5, 0.5, 0.5]),
                             radius=0.1, n_points=100):

    angles = np.random.uniform(0, 2*np.pi, n_points)
    z = np.random.uniform(-1, 1, n_points)
    r = np.sqrt(1 - z**2)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    sphere_points = np.column_stack((x, y, z))
    scaled_points = sphere_points * radius
    return center + scaled_points

###############################################################################
### Main                                                                    ###
###############################################################################
def main():
    index = 0

    gr_arr = np.array([1,2,3,4,6,8,16,24,32])

    # Test case 1: Standard set
    xyzset = create_standard_xyzset()
    create_test_case("standard", index, xyzset, gr_arr)
    index += 1

    # random tests
    for i in range(16):
        xyzset = create_random_xyzset()
        create_test_case("random", index, xyzset, gr_arr)
        index += 1

    # clustered dataset
    cluster_center = np.array([0.5, 0.5, 0.5])
    xyzset = create_clustered_xyzset(cluster_center)
    create_test_case("clustered", index, xyzset, gr_arr)
    index += 1

    # lattice (between 0,1 exclusive)
    xyzset = create_lattice_xyzset()
    create_test_case("lattice", index, xyzset, gr_arr)
    index += 1

    # Fibonacci sphere
    for N in [8,24,64,128]:
        xyzset = create_fibonacci_sphere_xyzset(N)
        create_test_case("fibonacci_sphere({})".format(N),
                         index, xyzset, gr_arr)
        index += 1
    
    # problematic tests due to voro++
    if 0:

        # Sparse data
        xyzset = create_sparse_xyzset(num_points=100, sparsity_level=0.95)
        create_test_case("sparse_data", index, xyzset, gr_arr)
        index += 1

        # Two points separated by epsilon
        xyzset = create_two_points_epsilon_xyzset()
        create_test_case("two_points_epsilon", index, xyzset, gr_arr)
        index += 1

        # Degenerate cases
        xyzset = create_degenerate_xyzset()
        create_test_case("degenerate_cases", index, xyzset, gr_arr)
        index += 1

        # Outliers
        base_set = create_standard_xyzset()
        xyzset = create_outliers_xyzset(base_set, outlier_factor=10)
        create_test_case("outliers", index, xyzset, gr_arr)
        index += 1

    # Noisy data
    base_set = create_lattice_xyzset()
    xyzset = create_noisy_xyzset(base_set, noise_level=0.1)
    create_test_case("noisy_data", index, xyzset, gr_arr)
    index += 1

    # Imbalanced classes
    xyzset = create_imbalanced_class_xyzset(majority_class_size=1000,
                                            minority_class_size=10)
    create_test_case("imbalanced_classes", index, xyzset, gr_arr)
    index += 1

    # Collinear points
    xyzset = create_collinear_xyzset()
    create_test_case("collinear", index, xyzset, gr_arr)
    index += 1

    # Concentric points
    xyzset = create_concentric_xyzset()
    create_test_case("concentric", index, xyzset, gr_arr)
    index += 1
    
    return

if __name__ == "__main__": 
    create_prefix()
    main()
    create_suffix()

###############################################################################
### End                                                                     ###
###############################################################################
