
#include <arguments.hpp>
#include <xyzset.hpp>
#include <status.hpp>

#include <knn.hpp>
#include <dnn.hpp>

#include <iostream>
#include <chrono>

#include <vector>
#include <array>
#include <cstdint>
#include <thread>
#include <mutex>
#include <iomanip>
#include <fstream>

#include <thread>

#define FP_INFINITY 128.00f

#define USE_NEW_TESSELLATE 1

namespace votess {

///////////////////////////////////////////////////////////////////////////////
/// Struct Votess Arguments                                                 ///
///////////////////////////////////////////////////////////////////////////////

struct vtargs {

  struct args::global global;
  struct args::xyzset xyzset;
  struct args::knn knn;
  struct args::cc cc;

  const size_t nthreads;

  vtargs(
    const int _k,
    const int _grid_resolution = 1,
    const int _nthreads = 1,
    const int _p_maxsize = ARGS_DEFAULT_P_MAXSIZE,
    const int _t_maxsize = ARGS_DEFAULT_T_MAXSIZE) 
    : global(_k),
      xyzset(_grid_resolution),
      knn(_k, _grid_resolution),
      cc(_k, _p_maxsize , _t_maxsize),
      nthreads(_nthreads) {}

  void set_k(const int k) {
    global.k = k;
    knn.k = k;
    cc.k = k;
  }

};


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

/* ------------------------------------------------------------------------- */
/* Dnn Class                                                                 */
/* ------------------------------------------------------------------------- */

template <typename Ti>
class dnn {

  public:
    class proxy;

    dnn();
    dnn(std::vector<Ti>& _list, std::vector<Ti>& _offs);
    dnn(std::vector<Ti>&& _list, std::vector<Ti>&& _offs);

    const proxy operator[](const int i) const;
    proxy operator[](const int i);
    size_t size() const;

    void print() const;
    void savetxt(const std::string& fname) const;

    std::vector<Ti> list;
    std::vector<Ti> offs;

  private:

};

template<typename Ti>
dnn<Ti>::dnn() {
  internal::check_sinteger<Ti>();
}

template<typename Ti>
dnn<Ti>::dnn(std::vector<Ti>& _list, std::vector<Ti>& _offs)
: list(_list), offs(_offs) {
  internal::check_sinteger<Ti>();
}

template<typename Ti>
dnn<Ti>::dnn(std::vector<Ti>&& _list, std::vector<Ti>&& _offs)
: list(std::move(_list)), offs(std::move(_offs)) {
  internal::check_sinteger<Ti>();
}


template<typename Ti>
const typename dnn<Ti>::proxy dnn<Ti>::operator[](const int i) const {
  return proxy(list, offs, i);
}

template<typename Ti>
typename dnn<Ti>::proxy dnn<Ti>::operator[](const int i) {
  return proxy(list, offs, i);
}

template<typename Ti>
size_t dnn<Ti>::size() const {
  return offs.size() - 1;
}

template <typename Ti>
void dnn<Ti>::print() const {
  if (this->list.empty()) {
    std::cout<<"{}"<<std::endl;
    return;
  }

  Ti index = 0;
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

template <typename Ti>
void dnn<Ti>::savetxt(const std::string& fname) const {
  std::ofstream fp(fname);
  if (!fp) {
    std::cerr<<"Failed to open file: "<<fname<<std::endl;
    return;
  }

  Ti index = 0;
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

/* ------------------------------------------------------------------------- */
/* Proxy Class                                                               */
/* ------------------------------------------------------------------------- */

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


///////////////////////////////////////////////////////////////////////////////
/// Tesellate internal functions                                            ///
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
/// Votess Internal Functions                                               ///
///////////////////////////////////////////////////////////////////////////////

namespace dtessellate {

/* ------------------------------------------------------------------------- */
/* CPU Implementation                                                        */
/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
class dnn<Ti>
cpu(
  std::vector<std::array<Tf,3>>& inset,
  const struct vtargs& args
) {

  const auto pair = xyzset::sort<Ti,Tf>(inset, args.xyzset);
  const auto& id       = pair.first;
  const auto& offset   = pair.second;

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
  
  const size_t nthreads = std::thread::hardware_concurrency(); 

  const size_t chunksize  = refsize / nthreads;

  std::vector<std::thread> threads(nthreads);
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

/* ------------------------------------------------------------------------- */
/* SYCL Implementation                                                       */
/* ------------------------------------------------------------------------- */

template <typename Ti, typename Tf>
class dnn<Ti>
gpu(
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

} // namespace dtessellate 


///////////////////////////////////////////////////////////////////////////////
/// Votess Tesellate Function                                               ///
///////////////////////////////////////////////////////////////////////////////

template <typename Ti, typename Tf>
class dnn<Ti>
#if USE_NEW_TESSELLATE
newtesellate(
#else
tesellate(
#endif
  std::vector<std::array<Tf,3>>& xyzset,
  struct vtargs args,
  const enum device device
) {

  static_assert(std::is_integral<Ti>::value && std::is_signed<Ti>::value,
    "Template type Ti must be a signed integer type."
  );

  static_assert(std::is_floating_point<Tf>::value,
  "Template type Tf must be a floating-point type."
  );

  switch (device) {
    case (device::cpu): return dtessellate::cpu<Ti,Tf>(xyzset, args);
    case (device::gpu): return dtessellate::gpu<Ti,Tf>(xyzset, args);
  }

  class dnn<Ti> err;
  return err;
  
}

///////////////////////////////////////////////////////////////////////////////
/// New Tesellate Functions                                                 ///
///////////////////////////////////////////////////////////////////////////////

template <typename Ti>
void 
tmpnn_fill(
  std::vector<std::vector<Ti>>& tmpnn,
  const std::vector<Ti>& indices,
  const std::vector<cc::state>& states,
  const std::vector<Ti>& knn,
  const struct vtargs& args
) {

  const auto& k = args.knn.k;

  for (size_t i = 0; i < indices.size(); i++) {
    const auto& index = indices[i];

//  if (!states[i].get(cc::security_radius_reached)) {
//    continue;
//  }

    tmpnn[index].resize(k);
    std::copy(knn.begin() + k * (i + 0),
              knn.begin() + k * (i + 1),
              tmpnn[index].begin());

  }

}

template <typename Ti>
class dnn<Ti>
tmpnn_getdnn(std::vector<std::vector<Ti>>& tmpnn) {

  const Ti xyzsize =  tmpnn.size();
  std::vector<Ti> _list(0);
  std::vector<Ti> _offs(xyzsize + 1);
  _offs[0] = 0;

  for (size_t i = 0; i < xyzsize; i++) {

    for (const auto& neighbor : tmpnn[i]) {
      if (neighbor == cc::k_undefined) {
        break;
      }
      _list.push_back(neighbor);
      _offs[i+1] += 1;
    }
    
    tmpnn[i].clear();
    tmpnn[i].shrink_to_fit();

  }

  for (size_t i = 1; i < xyzsize + 1; i++) {
    _offs[i] += _offs[i - 1];
  }

  return dnn(std::move(_list), std::move(_offs));

}

template <typename Ti, typename Tf>
void
__cpu__tesellate(

  const std::vector<std::array<Tf,3>>& xyzset,
  const std::vector<Ti>& id,
  const std::vector<Ti>& offset,

  std::vector<std::vector<Ti>>& tmpnn,
  const std::vector<Ti>& indices,
  std::vector<cc::state>& states, 

  const struct vtargs& args

) {
  
  const Ti xyzsize = xyzset.size();
  const Ti subsize = indices.size();

  const auto& k = args.global.k;
  const auto& p_maxsize = args.cc.p_maxsize;
  const auto& t_maxsize = args.cc.t_maxsize;

  std::vector<Ti> heap_id(subsize * k, 0);
  std::vector<Tf> heap_pq(subsize * k, FP_INFINITY);
  std::vector<Ti>& knn = heap_id;

  std::vector<Ti> dknn(subsize * k, __INTERNAL__K_UNDEFINED);

  std::vector<Tf>       P(subsize * p_maxsize * 4);
  std::vector<uint8_t>  T(subsize * t_maxsize * 3);
  std::vector<uint8_t> dR(subsize * p_maxsize);

  const size_t nthreads = std::thread::hardware_concurrency(); 
  const size_t chunksize  = subsize / nthreads;

  std::vector<std::thread> threads(nthreads);
  for (size_t i = 0; i < nthreads; i++) {

    const size_t _start = i * chunksize;
    const size_t _end = (i == nthreads - 1) ? subsize : _start + chunksize;

    threads[i] = std::thread([&,_start,_end]() {
      for (size_t idx = _start; idx < _end; idx++) {
        knni::compute<Ti,Tf>(
          idx, indices[idx], 
          xyzset, xyzsize, id, offset, 
          xyzset, subsize,
          heap_id, heap_pq,
          args.knn
        );

        dnni::compute<Ti, Tf, uint8_t>( 
          idx, indices[idx],
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

  tmpnn_fill(tmpnn, indices, states, knn, args);

}


template <typename Ti, typename Tf>
class dnn<Ti>

#if USE_NEW_TESSELLATE
tesellate(
#else
newtesellate(
#endif
  std::vector<std::array<Tf,3>>& xyzset,
  struct vtargs args,
  const enum device device
) {

  static_assert(std::is_integral<Ti>::value && std::is_signed<Ti>::value,
    "Template type Ti must be a signed integer type."
  );

  static_assert(std::is_floating_point<Tf>::value,
  "Template type Tf must be a floating-point type."
  );

  const auto [id,offset] = xyzset::sort<Ti,Tf>(xyzset, args.xyzset);
  const size_t xyzsize = xyzset.size();
  const auto k0 = args.global.k;

  // TODO : Make errors actually good
  if (!xyzset::validate_xyzset<Tf>(xyzset)) {
    std::cerr<<"oops1"<<std::endl;
  }
  if (!xyzset::validate_id<Ti>(id)) {
    std::cerr<<"oops2"<<std::endl;
  }
  if (!xyzset::validate_offset<Ti>(offset)) {
    std::cerr<<"oops3"<<std::endl;
  }
  
  std::vector<std::vector<Ti>> tmpnn(xyzsize);

  const int chunksize = 10000;
  const int nruns = chunksize < xyzsize ? xyzsize / chunksize : 1;

  for (int run = 0; run < nruns; run++) {

    std::cout << "[data] run : " << run << " / " << nruns << std::endl;

    const size_t _start = run * chunksize;
    const size_t _end = (run == nruns - 1) ? xyzsize : _start + chunksize;

    size_t subsize = _end - _start;
    auto& k = args.global.k;

    std::vector<struct cc::state> states(subsize);
    std::vector<Ti> indices(subsize);

    for (size_t i = 0; i < subsize; i++) {
      indices[i] = _start + i;
    }

    args.set_k(k0);

    __cpu__tesellate(xyzset, id, offset, tmpnn, indices, states, args);
    
    while (1) {

      size_t cur = 0; 
      for (size_t i = 0; i < subsize; i++) {
        if (!states[i].get(cc::security_radius_reached)) {
          indices[cur++] = indices[i];
        }
      }
      subsize = cur;

      states.resize(subsize);
      for (size_t i = 0; i < subsize; i++) states[i].reset();

      if (subsize <= 0) {
        break;
      }
   
      args.set_k(k * 2);
      if (k > (xyzsize - 1)) {
        args.set_k(xyzsize - 1);
      }

      std::cout << "\t[data] recomputing with"
                << "\tk : " << k 
                << "\tsize : " << subsize 
                << std::endl;

      __cpu__tesellate(xyzset, id, offset, tmpnn, indices, states, args);

      if (k >= (xyzsize - 1)) {
        break;
      }

    }

  }


  return tmpnn_getdnn(tmpnn);

}

///////////////////////////////////////////////////////////////////////////////
/// End                                                                     ///
///////////////////////////////////////////////////////////////////////////////
} // namespace votess
