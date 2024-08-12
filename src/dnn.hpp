
namespace votess {

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

}

///////////////////////////////////////////////////////////////////////////////
/// class dnn                                                               ///
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
/// class proxy                                                             ///
///////////////////////////////////////////////////////////////////////////////

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
