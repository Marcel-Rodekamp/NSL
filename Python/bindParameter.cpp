#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL;

namespace NSL::Python {
    class PyParameter : public Parameter {
    private:
        py::dict dictionary_;
    
    public:
        using Parameter::Parameter;

        PyParameter(py::dict dictionary)  : dictionary_(dictionary) {
            for (std::pair<py::handle, py::handle> item : dictionary){
                std::string key = item.first.cast<std::string>();
                py::handle value = item.second;
                if (py::isinstance<py::int_>(value)) {
                    addParameter<int>(key, value.cast<int>());
                } else if (py::isinstance<py::float_>(value)) {
                    addParameter<double>(key, value.cast<double>());
                } else if (py::isinstance<py::str>(value)) {
                    addParameter<std::string>(key, value.cast<std::string>());
                } else if (py::type::of(value).is(py::module::import("builtins").attr("complex"))) {
                    addParameter<NSL::complex<double>>(key, value.cast<NSL::complex<double>>());
                } else if (py::isinstance<py::module>(value)) {
                    throw std::invalid_argument("Not implemented yet");
                } else if (py::isinstance<SpatialLattice<float>>(value)) {
                    addParameter<SpatialLattice<float>>(key, value.cast<SpatialLattice<float>>());
                    std::cout << "SpatialLattice" << std::endl;
                } else {
                    py::str type_str = py::str(value.get_type().attr("__name__"));
                    std::string type_name = type_str.operator std::string();
                    throw std::invalid_argument("Unsupported type: " + type_name);
                }
            }
        }

        py::object getDictionary() {
            return dictionary_;
        }

        void setDictionary(py::dict dictionary) {
            dictionary_ = dictionary;
            *this = PyParameter(dictionary_);
        }

        void updateMap(){
            *this = PyParameter(dictionary_);
        }
    };

    void bindParameter(py::module &m) {
        py::class_<PyParameter>(m, "Parameter")
            .def(py::init<>())
            .def(py::init<py::dict>())
            .def_property("p", &PyParameter::getDictionary, &PyParameter::setDictionary)
            .def("updateMap", &PyParameter::updateMap);
    }
}
