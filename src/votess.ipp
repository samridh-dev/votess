#include <arguments.hpp>
#include <xyzset.hpp>
#include <status.hpp>

#include <knn.hpp>
#include <dnn.hpp>

#include <iostream>
#include <chrono>

#include <vector>
#include <thread>
#include <mutex>
#include <iomanip>
#include <fstream>

#include <thread>

#define FP_INFINITY 128.00f

namespace votess {

///////////////////////////////////////////////////////////////////////////////
/// Internal functions
///////////////////////////////////////////////////////////////////////////////

namespace internal {

  template <typename T> 
  void check_sinteger() {
    static_assert(std::is_signed<T>::value && std::is_integral<T>::value,
    "Template type T must be a signed integer type."
    );
  }

}

///////////////////////////////////////////////////////////////////////////////
/// Direct neighbor implementation
///////////////////////////////////////////////////////////////////////////////

/*  ------------------------------------------------------------------------ */
/// Proxy Class
/*  ------------------------------------------------------------------------ */

template <typename T>
class dnn<T>::proxy {
  public:
    proxy(std::vector<T>& list, std::vector<T>& offs, const size_t index);
    T& operator[](const size_t i);
    size_t size() const;
  private:
    std::vector<T>& _list;
    std::vector<T>& _offs;
    size_t _index;
};

template<typename T>
dnn<T>::proxy::proxy( std::vector<T>& list,
  std::vector<T>& offs,
  const size_t index
) : _list(list), _offs(offs), _index(index) {}

template<typename T>
T& dnn<T>::proxy::operator[](const size_t i) {
  return _list[_offs[_index] + i];
}

template<typename T>
size_t dnn<T>::proxy::size() const {
  if (_index == _offs.size() - 1) return _list.size() - _offs[_index];
  else return _offs[_index + 1] - _offs[_index];
}

/*  ------------------------------------------------------------------------ */
/// Dnn Class
/*  ------------------------------------------------------------------------ */

/*  ------------------------------------------------------------------------ */

template<typename T>
dnn<T>::dnn() {
  internal::check_sinteger<T>();
}

/*  ------------------------------------------------------------------------ */

template<typename T>
dnn<T>::dnn(std::vector<T>& _list, std::vector<T>& _offs)
: list(_list), offs(_offs) {
  internal::check_sinteger<T>();
}

/*  ------------------------------------------------------------------------ */

template<typename T>
const typename dnn<T>::proxy dnn<T>::operator[](const int i) const {
  return proxy(list, offs, i);
}

/*  ------------------------------------------------------------------------ */

template<typename T>
typename dnn<T>::proxy dnn<T>::operator[](const int i) {
  return proxy(list, offs, i);
}

/*  ------------------------------------------------------------------------ */

template<typename T>
size_t dnn<T>::size() const {
  return offs.size() - 1;
}

/*  ------------------------------------------------------------------------ */

template <typename T>
void dnn<T>::print() const {
  if (this->list.empty()) {
    std::cout<<"{}"<<std::endl;
    return;
  }

  T index = 0;
  size_t counter = 1;
  std::cout<<"{\n  {";
  for (auto i : this->list) {
    if (index == this->offs[counter]) {
      if (counter < this->offs.size() - 1) {
        counter += 1;
      }
      std::cout << "}\n  {";
    }

    std::cout<<std::setw(1)<<i<<", ";
    index += 1;
  }
  std::cout<<"\n}"<<std::endl;;
}

/*  ------------------------------------------------------------------------ */

template <typename T>
void dnn<T>::savetxt(const std::string& fname) const {
  std::ofstream fp(fname);
  if (!fp) {
    std::cerr<<"Failed to open file: "<<fname<<std::endl;
    return;
  }

  T index = 0;
  size_t counter = 1;
  for (auto i : this->list) {
    if (index == this->offs[counter]) {
      if (counter < this->offs.size() - 1) {
        counter += 1;
      }
      fp<<"\n";
    }
    fp<<std::setw(1)<<i<<" ";
    index += 1;
  }
  fp<<"\n";
  fp.close();
}

/*  ------------------------------------------------------------------------ */

///////////////////////////////////////////////////////////////////////////////
/// Tesellate helper function
///////////////////////////////////////////////////////////////////////////////

static inline void 
print_device(const sycl::queue& queue) {

  const auto d = queue.get_device();
  
  const size_t gibibyte = std::pow(2,30);
  const size_t kibibyte = std::pow(2,10);

  std::cout << " :: DEVICE INFO :: " << "\n";

  std::cout << " :: Using Device: " 
            << d.get_info<sycl::info::device::name>() << "\n";
  
  std::cout << " :: Device Info: \n";

  std::cout << "    -- Vendor: "  
            << d.get_info<sycl::info::device::vendor>() 
            << "\n";

  std::cout << "    -- Device Type: " 
            << (d.is_cpu() ? "CPU" : (d.is_gpu() ? "GPU" : "Other")) 
            << "\n";

  std::cout << "    -- Maximum Work Group Size: " 
            << d.get_info<sycl::info::device::max_work_group_size>() 
            << "\n";

  std::cout << "    -- Preferred Vector Width for Double: " 
            << d.get_info<sycl::info::device::preferred_vector_width_double>() 
            << "\n";

  std::cout << "    -- Max Compute Units: " 
            << d.get_info<sycl::info::device::max_compute_units>() 
            << "\n";

  std::cout << "    -- Max Memory Allocation Size: " 
            << d.get_info<sycl::info::device::max_mem_alloc_size>() / gibibyte
            << " GiB"
            << "\n";

  std::cout << "    -- Global Memory Size: " 
            << d.get_info<sycl::info::device::global_mem_size>() / gibibyte
            << " GiB"
            << "\n";
 
  std::cout << "    -- Local Memory Size: " 
            << d.get_info<sycl::info::device::local_mem_size>() / kibibyte
            << " KiB"
            << "\n";
  
  std::cout << " :: INFO END :: " << std::endl;

}

///////////////////////////////////////////////////////////////////////////////
/// Tesellate Function
///////////////////////////////////////////////////////////////////////////////

template <typename Ti, typename Tf>
class dnn<Ti>
tesellate(
  std::vector<std::array<Tf,3>>& xyzset,
  const struct vtargs& args,
  const enum device device
) {

  static_assert(std::is_floating_point<Tf>::value,
  "Template type Tf must be a floating-point type."
  );

  switch (device) {
    case (device::cpu): return dtessellate::cpu<Ti,Tf>(xyzset, args);
    case (device::gpu): return dtessellate::gpu<Ti,Tf>(xyzset, args);
  }

  class dnn<Ti> failure;
  return failure;
}

/*  ------------------------------------------------------------------------ */
/// CPU Implementation
/*  ------------------------------------------------------------------------ */

template <typename Ti, typename Tf>
class dnn<Ti>
dtessellate::cpu(
  std::vector<std::array<Tf,3>>& inset,
  const struct vtargs& args
) {

  const auto sort_pair = xyzset::sort<Ti,Tf>(inset, args.xyzset);
  const auto& id       = sort_pair.first;
  const auto& offset   = sort_pair.second;

  if (!xyzset::validate_xyzset<Tf>(inset)) {
    std::cerr<<"oops1"<<std::endl;
  }
  if (!xyzset::validate_id<Ti>(id)) {
    std::cerr<<"oops2"<<std::endl;
  }
  if (!xyzset::validate_offset<Ti>(offset)) {
    std::cerr<<"oops3"<<std::endl;
  }

  const std::vector<std::array<Tf,3>>& xyzset = inset;

  const size_t xyzsize = xyzset.size();
  const size_t refsize = xyzsize;

  std::vector<Ti> heap_id(inset.size() * (args.knn.k + 0), 0);
  std::vector<Tf> heap_pq(inset.size() * (args.knn.k + 0), FP_INFINITY);
  std::vector<Ti>& knn = heap_id;

  std::vector<cc::state> states(xyzset.size());
  std::vector<Ti> dknn(refsize * args.cc.k, __INTERNAL__K_UNDEFINED);

  std::vector<Tf>       P(xyzsize * args.cc.p_maxsize * 4);
  std::vector<uint8_t>  T(xyzsize * args.cc.t_maxsize * 3);
  std::vector<uint8_t> dR(xyzsize * args.cc.p_maxsize);
  
  const size_t nthreads = args.nthreads > std::thread::hardware_concurrency() ? 
                          std::thread::hardware_concurrency() : args.nthreads;
  const size_t chunksize  = refsize / nthreads;
  std::vector<std::thread> threads(nthreads);

#if true
  for (size_t i = 0; i < nthreads; i++) {

    const size_t _start = i * chunksize;
    const size_t _end = (i == nthreads - 1) ? refsize : _start + chunksize;

    threads[i] = std::thread([&,_start,_end]() {
      for (size_t idx = _start; idx < _end; idx++) {
        knni::compute<Ti,Tf>(
          idx, 
          xyzset, xyzsize, id, offset, 
          xyzset, refsize,
          heap_id, heap_pq,
          args.knn
        );

        dnni::compute<Ti, Tf, uint8_t>( 
          idx,
          states, 
          P.data(), T.data(), dR.data(),
          knn, dknn,
          xyzset, xyzsize,
          xyzset, xyzsize,
          args.cc
        );
      }
    });

  }

  for (auto& thread : threads) thread.join();
#else
  for (size_t i = 0; i < nthreads; i++) {
    const size_t _start = i * chunksize;
    const size_t _end = (i == nthreads - 1) ? refsize : _start + chunksize;

    for (size_t idx = _start; idx < _end; idx++) {
      knni::compute<Ti, Tf>(
        idx,
        xyzset, xyzsize, id, offset,
        xyzset, refsize,
        heap_id, heap_pq,
        args.knn
      );

      dnni::compute<Ti, Tf, uint8_t>(
        idx,
        states,
        P.data(), T.data(), dR.data(),
        knn, dknn,
        xyzset, xyzsize,
        xyzset, xyzsize,
        args.cc
      );
    }
  }
#endif

  std::vector<Ti> _list(0);
  std::vector<Ti> _offs(refsize + 1);
  _offs[0] = 0;
  
  for (size_t n_i = 0; n_i < refsize; n_i++) {
    for (size_t k_i = 0; k_i < args.cc.k; k_i++) {
      const Ti neighbor = knn[n_i * args.cc.k + k_i];
      if (neighbor == cc::k_undefined) break;
      _list.push_back(neighbor);
      _offs[n_i + 1] += 1;
    }
  }
  for (size_t n_i = 1; n_i < refsize + 1; n_i++) {
    _offs[n_i] += _offs[n_i - 1];
  }

  class dnn<Ti> dnn(_list,_offs);

  return dnn;

}

/*  ------------------------------------------------------------------------ */
/// SYCL Implementation
/*  ------------------------------------------------------------------------ */

template <typename Ti, typename Tf>
class dnn<Ti>
dtessellate::gpu(
  std::vector<std::array<Tf,3>>& inset, 
  const struct vtargs& args
) {
  auto total_beg = std::chrono::high_resolution_clock::now();
  
  // get id, offset, and sort array
  const auto [id, offset] = xyzset::sort<Ti,Tf>(inset, args.xyzset);
  if (!xyzset::validate_xyzset<Tf>(inset)) {
    std::cout<<"oops1"<<std::endl;
  }
  if (!xyzset::validate_id<Ti>(id)) {
    std::cout<<"oops2"<<std::endl;
  }
  if (!xyzset::validate_offset<Ti>(offset)) {
    std::cout<<"oops3"<<std::endl;
  }

  std::vector<Tf> xyzset(3 * inset.size());
  for (size_t i = 0; i < inset.size(); i++) {
    xyzset[inset.size() * 0 + i] = inset[i][0];
    xyzset[inset.size() * 1 + i] = inset[i][1];
    xyzset[inset.size() * 2 + i] = inset[i][2];
  }

  const size_t xyzsize = xyzset.size() / 3;
  const size_t refsize = xyzsize;

  sycl::buffer<Tf,1> bxyzset(xyzset.data(), sycl::range<1>(xyzset.size()));
  sycl::buffer<Ti,1> boffset(offset.data(), sycl::range<1>(offset.size()));
  sycl::buffer<Ti,1> bid(id.data(), sycl::range<1>(id.size()));

  sycl::buffer<Ti,1> bheap_id(sycl::range<1>(refsize * (args.knn.k + 0)));
  sycl::buffer<Tf,1> bheap_pq(sycl::range<1>(refsize * (args.knn.k + 0)));

  sycl::buffer<cc::state,1> bstates(sycl::range<1>(xyzset.size() / 3));
  sycl::buffer<Ti,1>      bdknn(sycl::range<1>(refsize * args.cc.k));
  sycl::buffer<Tf,1>      bP(sycl::range<1>(refsize * args.cc.p_maxsize * 4));
  sycl::buffer<uint8_t,1> bT(sycl::range<1>(refsize * args.cc.t_maxsize * 3));
  sycl::buffer<uint8_t,1> bdR(sycl::range<1>(refsize * args.cc.p_maxsize));

  sycl::queue queue;
  print_device(queue);

  queue.submit([&](sycl::handler& cgh) {
    auto aheap_id = sycl::accessor(bheap_id, cgh, sycl::read_write);
    cgh.fill(aheap_id, 0);
  });
  queue.submit([&](sycl::handler& cgh) {
    auto aheap_pq = sycl::accessor(bheap_pq, cgh, sycl::read_write);
    cgh.fill(aheap_pq, FP_INFINITY);
  });
  queue.submit([&](sycl::handler& cgh) {
    auto adknn = sycl::accessor(bdknn, cgh, sycl::read_write);
    cgh.fill(adknn, __INTERNAL__K_UNDEFINED);
  });

  auto beg = std::chrono::high_resolution_clock::now();
  queue.submit([&](sycl::handler& cgh) {

    using namespace sycl;

    auto aheap_id = sycl::accessor(bheap_id, cgh, read_write);
    auto aheap_pq = sycl::accessor(bheap_pq, cgh, read_write);
    auto adknn = sycl::accessor(bdknn, cgh, read_write);
    auto aknn = aheap_id;

    auto axyzset = sycl::accessor(bxyzset, cgh, read_only);
    auto aoffset = sycl::accessor(boffset, cgh, read_only);
    auto aid = accessor(bid, cgh, read_only);
    
    auto aP  = sycl::accessor(bP,  cgh, sycl::read_write, property::no_init());
    auto aT  = sycl::accessor(bT,  cgh, sycl::read_write, property::no_init());
    auto adR = sycl::accessor(bdR, cgh, sycl::read_write, property::no_init());
    auto astates = sycl::accessor(bstates, cgh, sycl::read_write, 
                                  property::no_init());

    auto aargs_knn = args.knn;
    auto aargs_cc = args.cc;

    sycl::range<1> global_range(refsize);
    sycl::range<1> local_range(args.nthreads);
    sycl::nd_range<1> nd_range(global_range, local_range);
    
    cgh.parallel_for<class __sycl__main>(nd_range, [=](sycl::nd_item<1> item) {
      knni::compute<Ti,Tf>(
        item.get_global_id()[0], 
        axyzset, xyzsize, aid, aoffset, 
        axyzset, refsize,
        aheap_id, aheap_pq,
        aargs_knn
      );
      dnni::compute<Ti, Tf, uint8_t>( 
        item.get_global_id()[0], 
        astates, 
        aP, aT, adR,
        aknn, adknn,
        axyzset, xyzsize,
        axyzset, xyzsize,
        aargs_cc
      );
    });
    queue.wait();

  });

  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
  std::cout << "tesellation time: " 
    << time.count() / (1000.00f * 1000.00f) 
    << " s" << std::endl;

  beg = std::chrono::high_resolution_clock::now();
  auto hknn = bheap_id.get_host_access();
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
  std::cout << "knn transfer time: " 
    << time.count() / (1000.00f * 1000.00f) 
    << " s" << std::endl;

  beg = std::chrono::high_resolution_clock::now();
  std::vector<Ti> _list(0);
  std::vector<Ti> _offs(refsize + 1);
  _offs[0] = 0;
  
  for (size_t n_i = 0; n_i < refsize; n_i++) {
    for (size_t k_i = 0; k_i < args.cc.k; k_i++) {
      const Ti neighbor = hknn[n_i * args.cc.k + k_i];
      if (neighbor == cc::k_undefined) break;
      _list.push_back(neighbor);
      _offs[n_i + 1] += 1;
    }
  }
  for (size_t n_i = 1; n_i < refsize + 1; n_i++) {
    _offs[n_i] += _offs[n_i - 1];
  }
  class dnn<Ti> dnn(_list,_offs);

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
  std::cout << "dnn processing time: " 
            << time.count() / (1000.00f * 1000.00f) 
            << " s" << std::endl;

  auto total_end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::microseconds>
         (total_end - total_beg);
  std::cout << "total tesellation time: " 
            << time.count() / (1000.00f * 1000.00f) 
            << " s" << std::endl;
  return dnn;

}

} // namespace votess
