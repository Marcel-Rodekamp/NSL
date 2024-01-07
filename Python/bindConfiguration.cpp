#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "Configuration/Configuration.tpp"

namespace py = pybind11;
using namespace pybind11::literals;


namespace NSL::Python{

template<typename Type>
void bindConfiguration(py::module &m, std::string class_name){
    py::class_<NSL::Configuration<Type>>(m, class_name.c_str())
        .def(py::init<>())
        .def("__setitem__", [](NSL::Configuration<Type> &self, const std::string &key, const NSL::Tensor<Type> &value) {
            self[key] = value;
        })
        .def("__getitem__", [](NSL::Configuration<Type>  &self, const std::string &key) {
            return self.at(key);
        })
        .def("__repr__", [](NSL::Configuration<Type> &self){
            std::stringstream ss;
            ss << self;
            return ss.str();
        });
}

void bindConfiguration(py::module &m){
    bindConfiguration<float>(m, "ConfigurationF");
    bindConfiguration<double>(m, "ConfigurationD");
    bindConfiguration<NSL::complex<float>>(m, "ConfigurationCF");
    bindConfiguration<NSL::complex<double>>(m, "Configuration");
}
} // NSL::Python
