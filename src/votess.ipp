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
/// Tesellate Internal Functions                                            ///
///////////////////////////////////////////////////////////////////////////////

// NOTE: This is the source of continuosly increasing memory for long runs
template <typename Ti>
static void 
tmpnn_fill(
  std::vector<std::vector<Ti>>& tmpnn,
  const std::vector<Ti>& indices, const size_t size,
  const std::vector<Ti>& knn,
  const class vtargs& args
) {
  for (size_t i = 0; i < size; i++) {
    
    const int k = args["k"];
    const auto& index = indices[i];
    tmpnn[index].clear();

    size_t nsize = 0;
    for (int j = 0; j < k; j++) {
      if (knn[k * i + j] == __INTERNAL__K_UNDEFINED) {
        break;
      }
      nsize++;
    }
  
    tmpnn[index].resize(nsize);
    
    for (size_t j = 0; j < nsize; j++) {
      const auto& ki = knn[k * i + j];
      if (ki == __INTERNAL__K_UNDEFINED) {
        continue;
      }
      tmpnn[index][j] = ki;
    }

  }

}

template <typename Ti>
static class dnn<Ti>
tmpnn_getdnn(std::vector<std::vector<Ti>>& tmpnn) {

  const Ti xyzsize =  tmpnn.size();
  std::vector<Ti> _list(0);
  std::vector<Ti> _offs(xyzsize + 1);
  _offs[0] = 0;

  for (Ti i = 0; i < xyzsize; i++) {

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

  for (Ti i = 1; i < xyzsize + 1; i++) {
    _offs[i] += _offs[i - 1];
  }

  return dnn(std::move(_list), std::move(_offs));

}

///////////////////////////////////////////////////////////////////////////////
/// GPU Tesellate                                                           ///
///////////////////////////////////////////////////////////////////////////////

#define USE_NEW_IMPL 1
template <typename Ti, typename Tf>
static void
__gpu__tesellate(

  const std::vector<std::array<Tf,3>>& _xyzset,
  const std::vector<Ti>& id,
  const std::vector<Ti>& offset,

  const std::vector<std::array<Tf,3>>& _refset,
  std::vector<std::vector<Ti>>& tmpnn,
  std::vector<cc::state>& states, 

  const struct vtargs& args

) {

  const Ti xyzsize = _xyzset.size();
  const Ti refsize = _refset.size();

  const int chunksize = args["use_chunking"].get<bool>() ? 
                        args["chunksize"] : refsize;

  const int nruns = chunksize < refsize ? 
                    refsize / chunksize + 1: 1;

  Ti subsize = chunksize < refsize ? chunksize : refsize;

  const int ndsize = args["gpu_ndsize"].get<int>() > 0 ?
                     args["gpu_ndsize"] : 1;
  (void)ndsize;

  const int k = args["k"];
  const int p_maxsize = args["cc_p_maxsize"];
  const int t_maxsize = args["cc_t_maxsize"];

  std::vector<Tf> xyzset(3 * xyzsize);
  for (size_t i = 0; i < xyzsize; i++) {
    xyzset[xyzsize * 0 + i] = _xyzset[i][0];
    xyzset[xyzsize * 1 + i] = _xyzset[i][1];
    xyzset[xyzsize * 2 + i] = _xyzset[i][2];
  }

  // will help me when I need to separate xyzset and refset
  auto& refset = _refset;


  sycl::buffer<Ti, 1> bindices((sycl::range<1>(subsize)));

  sycl::buffer<Tf,1> bxyzset(xyzset.data(), sycl::range<1>(xyzset.size()));
  sycl::buffer<Ti,1> boffset(offset.data(), sycl::range<1>(offset.size()));
  sycl::buffer<Ti,1> bid(id.data(), sycl::range<1>(id.size()));

  sycl::buffer<Ti,1> bheap_id(sycl::range<1>(subsize * k));
  sycl::buffer<Tf,1> bheap_pq(sycl::range<1>(subsize * k));

  sycl::buffer<cc::state,1> bstates((sycl::range<1>(xyzsize)));
  sycl::buffer<Ti,1>      bdknn(sycl::range<1>(subsize * k));
  sycl::buffer<Tf,1>      bP(sycl::range<1>(subsize * p_maxsize * 4));
  sycl::buffer<uint8_t,1> bT(sycl::range<1>(subsize * t_maxsize * 3));
  sycl::buffer<uint8_t,1> bdR(sycl::range<1>(subsize * p_maxsize));

  sycl::queue queue;
  print_device(queue);

  for (int run = 0; run < nruns; run++) {
    
    std::cout << "[chunking] run : " << run << "/" << nruns << std::endl;

    const size_t _cstart = run * chunksize;
    const size_t _cend = (run == nruns - 1) ? refsize : _cstart + chunksize;
    subsize = _cend - _cstart;
    

    queue.submit([&](sycl::handler& cgh) {
      auto a = sycl::accessor(bheap_id, cgh, sycl::write_only, sycl::no_init);
      cgh.parallel_for<class fill_heap_id>
      (sycl::range<1>(chunksize * k), [=](sycl::id<1> idx) { 
        a[idx] = 0; 
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto a = sycl::accessor(bheap_pq, cgh, sycl::write_only, sycl::no_init);
      cgh.parallel_for<class fill_heap_pq>
      (sycl::range<1>(chunksize * k), [=](sycl::id<1> idx) {
        a[idx] = FP_INFINITY;
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto a = sycl::accessor(bdknn, cgh, sycl::write_only, sycl::no_init);
      cgh.parallel_for<class fill_dknn>
      (sycl::range<1>(chunksize * k), [=](sycl::id<1> idx) {
        a[idx] = __INTERNAL__K_UNDEFINED;
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto a = sycl::accessor(bindices, cgh, sycl::write_only, sycl::no_init);
      cgh.parallel_for<class indices_init>
      (sycl::range<1>(subsize), [=](sycl::id<1> idx) {
        a[idx] = _cstart + idx[0];
      });
    });

    queue.wait();

    auto args_cc = args.get_cc();
    auto args_knn = args.get_knn();
    
    queue.submit([&](sycl::handler& cgh) {

      using namespace sycl;

      const size_t sgsize = 32; // subgroup size;
    
      auto aindices = sycl::accessor(bindices, cgh, read_only);
      auto axyzset = sycl::accessor(bxyzset, cgh, read_only);
      auto aoffset = sycl::accessor(boffset, cgh, read_only);
      auto aid = sycl::accessor(bid, cgh, read_only);
      
      // optimize this first
      auto aheap_id = sycl::accessor(bheap_id, cgh, read_write);
      auto aheap_pq = sycl::accessor(bheap_pq, cgh, read_write);
      auto adknn = sycl::accessor(bdknn, cgh, read_write);

      auto aP  = sycl::accessor(bP,  cgh, read_write, property::no_init());
      auto aT  = sycl::accessor(bT,  cgh, read_write, property::no_init());
      auto adR = sycl::accessor(bdR, cgh, read_write, property::no_init());
      auto astates = accessor(bstates, cgh, read_write, property::no_init());

      auto aargs_knn = args_knn;
      auto aargs_cc = args_cc;

#if USE_NEW_IMPL 
      cgh.parallel_for<class __sycl__tessellate>
      (sycl::nd_range<1>(sycl::range<1>(subsize), sycl::range<1>(sgsize)),
      [=](sycl::nd_item<1> it) {

        const size_t index = it.get_global_linear_id();

        knni::compute<Ti, Tf>(
          index, aindices[index],
          axyzset, xyzsize, aid, aoffset,
          axyzset, subsize,
          aheap_id, aheap_pq, subsize,
          aargs_knn
        );

        #if 0
        dnni::compute<Ti, Tf, uint8_t>(
          index, aindices[index],
          astates,
          aP, aT, adR,
          aheap_id, subsize,
          adknn,
          axyzset, xyzsize,
          axyzset, subsize,
          aargs_cc
        );
        #endif

#else

      cgh.parallel_for<class __sycl__tessellate>
      (sycl::range<1>(subsize), 
      [=](sycl::id<1> idx) {

       const size_t index = idx[0];
        knni::compute<Ti, Tf>(
          index, aindices[index],
          axyzset, xyzsize, aid, aoffset,
          axyzset, subsize,
          aheap_id, aheap_pq,
          aargs_knn
        );
        #if 0
        dnni::compute<Ti, Tf, uint8_t>(
          index, aindices[index],
          astates,
          aP, aT, adR,
          aheap_id, adknn,
          axyzset, xyzsize,
          axyzset, subsize,
          aargs_cc
        );
        #endif


#endif

      });
    });
    queue.wait();

    auto hindices = bindices.get_host_access();
    auto hstates = bstates.get_host_access();
    auto hknn = bheap_id.get_host_access();

    std::vector<Ti> indices(hindices.begin(), hindices.end());
    states = std::vector<cc::state>(hstates.begin(), hstates.end());

#if USE_NEW_IMPL
    std::vector<Ti> _knn(hknn.begin(), hknn.end());
    std::vector<Ti> knn(_knn.size());
    for (size_t si = 0; si < subsize; si++) {
      for (size_t ki = 0; ki < k; ki++) {
        knn[k * si + ki] = _knn[subsize * ki + si];
      }
    }
#else
    std::vector<Ti> knn(hknn.begin(), hknn.end());
#endif

    tmpnn_fill(tmpnn, indices, subsize, knn, args);

  }


}

///////////////////////////////////////////////////////////////////////////////
/// CPU Tesellate                                                           ///
///////////////////////////////////////////////////////////////////////////////

template <typename Ti, typename Tf>
static void
__cpu__tesellate(

  const std::vector<std::array<Tf,3>>& xyzset,
  const std::vector<Ti>& id,
  const std::vector<Ti>& offset,

  const std::vector<std::array<Tf,3>>& refset,
  std::vector<std::vector<Ti>>& tmpnn,
  std::vector<cc::state>& states, 

  const class vtargs& args

) {
  
  const Ti xyzsize = xyzset.size();
  const Ti refsize = refset.size();

  const size_t nthreads = args["cpu_nthreads"].get<size_t>() != 0 ?
                          args["cpu_nthreads"] : 
                          std::thread::hardware_concurrency(); 

  const int chunksize = args["use_chunking"].get<bool>() ? 
                        args["chunksize"] : refsize;

  const int nruns = chunksize < refsize ? 
                    refsize / chunksize + 1: 1;

  Ti subsize = chunksize < refsize ? chunksize : refsize;

  std::cout << "nthread = " << nthreads << std::endl; 
  std::cout << "chunksize = " << chunksize << std::endl; 
  std::cout << "nruns = " << nruns << std::endl; 

  const int k = args["k"];
  const int p_maxsize = args["cc_p_maxsize"];
  const int t_maxsize = args["cc_t_maxsize"];

  std::vector<Ti> indices(subsize);

  std::vector<Ti> heap_id(subsize * k, 0);
  std::vector<Tf> heap_pq(subsize * k, FP_INFINITY);
  std::vector<Ti>& knn = heap_id;

  std::vector<Ti> dknn(subsize * k, __INTERNAL__K_UNDEFINED);

  std::vector<Tf>       P(subsize * p_maxsize * 4);
  std::vector<uint8_t>  T(subsize * t_maxsize * 3);
  std::vector<uint8_t> dR(subsize * p_maxsize);

  for (int run = 0; run < nruns; run++) {

    std::fill(dknn.begin(), dknn.end(), __INTERNAL__K_UNDEFINED);
    std::fill(heap_id.begin(), heap_id.end(), 0);
    std::fill(heap_pq.begin(), heap_pq.end(), 
              std::numeric_limits<Tf>::infinity());

    const size_t threadsize  = subsize / nthreads;

    const size_t _cstart = run * chunksize;
    const size_t _cend = (run == nruns - 1) ? refsize : _cstart + chunksize;
    subsize = _cend - _cstart;

    for (Ti i = 0; i < subsize; i++) {
      indices[i] = _cstart + i;
    }

    const auto args_knn = args.get_knn();
    const auto args_cc  = args.get_cc();

    std::vector<std::thread> threads(nthreads);
    for (size_t i = 0; i < nthreads; i++) {

      const size_t _tstart = i * threadsize;
      const size_t _tend = (i != nthreads - 1) ? _tstart + threadsize 
                                               : subsize;

      threads[i] = std::thread([&,_tstart,_tend]() {
        for (size_t idx = _tstart; idx < _tend; idx++) {
          knni::compute<Ti,Tf>(
            idx, indices[idx], 
            xyzset, xyzsize, id, offset, 
            refset, subsize,
            heap_id, heap_pq,
            args_knn
          );

          dnni::compute<Ti, Tf, uint8_t>( 
            idx, indices[idx],
            states, 
            P.data(), T.data(), dR.data(),
            knn, dknn,
            xyzset, xyzsize,
            refset, subsize,
            args_cc
          );
        }
      });

    }
    for (auto& thread : threads) thread.join();

    tmpnn_fill(tmpnn, indices, subsize, knn, args);

  }

}

///////////////////////////////////////////////////////////////////////////////
/// CPU Recompute                                                           ///
///////////////////////////////////////////////////////////////////////////////

// NOTE: I decided to reimplement tesellation to recompute, not because I
//        could've done a DRY job, but because I needed to ensure memory
//        efficiency. I would not get that if I reused __cpu__tesellate.
//        I also thought of embedding recompute to the tesellate functions as
//        that could improve cache locality, but it would become an issue
//        maintaining memory usage. Heap fragmentation is a serious problem in
//        this program as in a single run, I have observed worsening
//        performance over time.

template <typename Ti, typename Tf>
static void
__cpu__recompute(

  const std::vector<std::array<Tf,3>>& xyzset,
  const std::vector<Ti>& id,
  const std::vector<Ti>& offset,

  const std::vector<std::array<Tf,3>>& refset,
  std::vector<std::vector<Ti>>& tmpnn,
  std::vector<cc::state>& states, 

  class vtargs args

) {

  const Ti xyzsize = xyzset.size();
  const Ti refsize = refset.size();

  const size_t nthreads = args["cpu_nthreads"].get<size_t>() != 0 ?
                          args["cpu_nthreads"] : 
                          std::thread::hardware_concurrency(); 

  Ti subsize = 0;
  for (auto& s : states) {
    if (!s.get(cc::security_radius_reached)) {
      subsize++;
    }
  }

  std::vector<Ti> indices(refsize);
  for (Ti i = 0, cur = 0; i < refsize; i++) {
    if (!states[i].get(cc::security_radius_reached)) {
      indices[cur++] = i;
    }
  } indices.resize(subsize);

  int k = args["k"];
  int p_maxsize = args["cc_p_maxsize"];
  int t_maxsize = args["cc_t_maxsize"];

  std::vector<Ti> heap_id(subsize * k, 0);
  std::vector<Tf> heap_pq(subsize * k, std::numeric_limits<Tf>::infinity());
  std::vector<Ti> dknn(subsize * k, __INTERNAL__K_UNDEFINED);
  std::vector<Ti>& knn = heap_id;

  std::vector<Tf>       P(subsize * p_maxsize * 4);
  std::vector<uint8_t>  T(subsize * t_maxsize * 3);
  std::vector<uint8_t> dR(subsize * p_maxsize);

  while (1) {

    bool update_t_maxsize = false;
    bool update_p_maxsize = false;
  
    Ti cur = 0;
    for (Ti i = 0; i < subsize; i++) {
      const auto index = indices[i];
      if (!states[index].get(cc::security_radius_reached)) {
        indices[cur++] = index;
      } 
      if (states[index].get(cc::error_p_overflow)) {
        update_p_maxsize = true;
      } 
      if (states[index].get(cc::error_t_overflow)) {
        update_t_maxsize = true;
      } 
    } 

    subsize = cur;
    for (Ti i = 0; i < refsize; i++) states[i].reset();

    if (subsize <= 0) {
      break;
    }

    k = (k * 2) > (refsize - 1) ? refsize - 1 : k * 2;
    p_maxsize *= update_p_maxsize ? 2 : 1;
    t_maxsize *= update_t_maxsize ? 2 : 1;

    args["k"] = k;
    args["cc_p_maxsize"] = p_maxsize;
    args["cc_t_maxsize"] = t_maxsize;

    heap_id.clear();
    heap_pq.clear();
    dknn.clear();
    P.clear();
    T.clear();
    dR.clear();

    heap_id.resize(subsize * k, 0);
    heap_pq.resize(subsize * k, std::numeric_limits<Tf>::infinity());
    dknn.resize(subsize * k, __INTERNAL__K_UNDEFINED);

    P.resize(subsize * p_maxsize * 4);
    T.resize(subsize * t_maxsize * 3);
    dR.resize(subsize * p_maxsize);

    const size_t threadsize  = subsize / nthreads;

    const auto args_knn = args.get_knn();
    const auto args_cc  = args.get_cc();

    std::vector<std::thread> threads(nthreads);
    for (size_t i = 0; i < nthreads; i++) {

      const size_t _tstart = i * threadsize;
      const size_t _tend = (i != nthreads - 1) ? _tstart + threadsize : subsize;

      threads[i] = std::thread([&,_tstart,_tend]() {
        for (size_t idx = _tstart; idx < _tend; idx++) {
          knni::compute<Ti,Tf>(
            idx, indices[idx], 
            xyzset, xyzsize, id, offset, 
            refset, subsize,
            heap_id, heap_pq,
            args_knn
          );

          dnni::compute<Ti, Tf, uint8_t>( 
            idx, indices[idx],
            states, 
            P.data(), T.data(), dR.data(),
            knn, dknn,
            xyzset, xyzsize,
            refset, subsize,
            args_cc
          );
        }
      });

    }
    for (auto& thread : threads) thread.join();

    tmpnn_fill(tmpnn, indices, subsize, knn, args);

    if (k >= (refsize - 1)) {
      break;
    }

  }

}

///////////////////////////////////////////////////////////////////////////////
/// Tesellate End Function                                                  ///
///////////////////////////////////////////////////////////////////////////////

template <typename Ti, typename Tf>
class dnn<Ti>
tesellate(
  std::vector<std::array<Tf,3>>& xyzset,
  class vtargs args,
  const enum device device
) {

  static_assert(std::is_integral<Ti>::value && std::is_signed<Ti>::value,
    "Template type Ti must be a signed integer type."
  );

  static_assert(std::is_floating_point<Tf>::value,
  "Template type Tf must be a floating-point type."
  );

  const auto [id,offset] = xyzset::sort<Ti,Tf>(xyzset, args.get_xyzset());

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

  const auto& refset = xyzset;
  const size_t refsize = refset.size();

  std::vector<std::vector<Ti>>  tmpnn(refsize);
  std::vector<struct cc::state> states(refsize);

  for (size_t i = 0; i < states.size(); i++) states[i].reset();

  switch (device) {
    case (device::gpu): 
      __gpu__tesellate(xyzset, id, offset, refset, tmpnn, states, args);
      break;
    case (device::cpu): 
      __cpu__tesellate(xyzset, id, offset, refset, tmpnn, states, args);
      break;
  }
  
  if (args["use_recompute"].get<bool>()) {
    std::cout << "recomputing" << std::endl;
    __cpu__recompute(xyzset, id, offset, refset, tmpnn, states, args);
  }

  int n_sr_nreached = 0;
  int n_inf_boundary = 0;
  int n_nvalid_vertices = 0;
  int n_nvalid_neighbor = 0;
  int n_p_overflow = 0;
  int n_t_overflow = 0;
  int n_error = 0;

  for (auto s: states) {
    if (!s.get(cc::security_radius_reached)) n_sr_nreached++;
    if (s.get(cc::error_infinite_boundary)) n_inf_boundary++;
    if (s.get(cc::error_nonvalid_vertices)) n_nvalid_vertices++;
    if (s.get(cc::error_nonvalid_neighbor)) n_nvalid_neighbor++;
    if (s.get(cc::error_p_overflow)) n_p_overflow++;
    if (s.get(cc::error_t_overflow)) n_t_overflow++;
    if (s.get(cc::error_occurred)) n_error++;
  }

  std::cerr << "[fail] " << "sradius : " << n_sr_nreached << "\n";
  std::cerr << "       " << "p overflow: " << n_p_overflow << "\n";
  std::cerr << "       " << "t overflow: " << n_t_overflow << "\n";
  std::cerr << "       " << "inf boundary: " << n_inf_boundary << "\n";
  std::cerr << "       " << "nvalid vertice: " << n_nvalid_vertices << "\n";
  std::cerr << "       " << "nvalid neighbor: " << n_nvalid_neighbor << "\n";
  std::cerr << "       " << "nerrors : " << n_error << "\n";

  return tmpnn_getdnn(tmpnn);

}

///////////////////////////////////////////////////////////////////////////////
/// End                                                                     ///
///////////////////////////////////////////////////////////////////////////////

} // namespace votess
