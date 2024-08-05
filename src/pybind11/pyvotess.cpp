#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "votess.hpp"
#include "arguments.hpp"

PYBIND11_MODULE(pyvotess, module) {
  module.doc() = "Python binding for votess";

  pybind11::enum_<votess::device>(module, "device")
    .value("cpu", votess::device::cpu)
    .value("gpu", votess::device::gpu)
    .export_values();

  pybind11::class_<votess::vtargref>(module, "vtargref")
    .def(pybind11::init<>());

  pybind11::class_<votess::vtargs>(module, "vtargs")
    .def(pybind11::init<>())
    .def("__getitem__", [](votess::vtargs &self, const std::string &key) 
        -> votess::vtargref & {
      return self[key];
    }, pybind11::return_value_policy::reference_internal)
    .def("__setitem__", [](votess::vtargs &self,
                           const std::string &key,
                           pybind11::object value) {
      if (pybind11::isinstance<pybind11::int_>(value)) {
        self[key] = value.cast<int>();
      } else if (pybind11::isinstance<pybind11::float_>(value)) {
        self[key] = value.cast<float>();
      } else if (pybind11::isinstance<pybind11::str>(value)) {
        self[key] = value.cast<std::string>();
      } else if (pybind11::isinstance<pybind11::bool_>(value)) {
        self[key] = value.cast<bool>();
      } else {
        throw std::invalid_argument("Unsupported value type");
      }
    });

  pybind11::class_<votess::dnn<int>::proxy>(module, "Proxy")
    .def("__getitem__", [](votess::dnn<int>::proxy &self, int i) -> int & {
      return self[i];
    }, pybind11::return_value_policy::reference_internal)
    .def("size", &votess::dnn<int>::proxy::size);

  pybind11::class_<votess::dnn<int>>(module, "dnn")
    .def(pybind11::init<>())
    .def(pybind11::init<std::vector<int>&, std::vector<int>&>())
    .def("size", &votess::dnn<int>::size)
    .def("__getitem__", [](votess::dnn<int> &instance, int i) 
        -> votess::dnn<int>::proxy {
      return instance[i];
    }, pybind11::return_value_policy::reference_internal)
    .def("print", &votess::dnn<int>::print)
    .def("savetxt", &votess::dnn<int>::savetxt);

  module.def(
    "tesellate",
    &votess::tesellate<int, float>,
    pybind11::arg("xyzset"),
    pybind11::arg("args"),
    pybind11::arg("device") = votess::device::gpu,
    "A function to tessellate XYZ data."
  );
}
