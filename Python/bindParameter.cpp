#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

enum class PyType {
    INT,
    FLOAT,
    STR,
    COMPLEX,
    MODULE,
    SPATIAL_LATTICE,
    UNKNOWN
};

PyType get_pytype(py::handle& pyvalue) {
    if (py::isinstance<py::int_>(pyvalue)) {
        return PyType::INT;
    } else if (py::isinstance<py::float_>(pyvalue)) {
        return PyType::FLOAT;
    } else if (py::isinstance<py::str>(pyvalue)) {
        return PyType::STR;
    } else if (py::type::of(pyvalue).is(py::module::import("builtins").attr("complex"))) {
        return PyType::COMPLEX;
    } else if (py::isinstance<py::module>(pyvalue)) {
        return PyType::MODULE;
    } else if (py::isinstance<SpatialLattice<float>>(pyvalue)) {
        return PyType::SPATIAL_LATTICE;
    } else {
        return PyType::UNKNOWN;
    }
}

namespace pybind11 {
namespace detail {

    // Empty Pybind11 type caster for converting Python dictionary to NSL::Parameter object
    template <>
    struct type_caster<NSL::Parameter> {
    public:
        PYBIND11_TYPE_CASTER(NSL::Parameter, _("NSL::Parameter"));

        // Conversion from Python to C++
        bool load(handle src, bool) {
            if (py::isinstance<py::dict>(src)) {
                py::dict src_dict = src.cast<py::dict>();
                NSL::Parameter param;
                for (std::pair<py::handle, py::handle> item : src_dict){
                    std::string key = item.first.cast<std::string>();
                    py::handle pyvalue = item.second;

                    switch(get_pytype(pyvalue)) {
                        case PyType::INT:
                            param.addParameter<NSL::size_t>(key, pyvalue.cast<int>());
                            break;
                        case PyType::FLOAT:
                            param.addParameter<float>(key, pyvalue.cast<double>());
                            break;
                        case PyType::STR:
                            param.addParameter<std::string>(key, pyvalue.cast<std::string>());
                            break;
                        case PyType::COMPLEX:
                            param.addParameter<NSL::complex<double>>(key, pyvalue.cast<NSL::complex<double>>());
                            break;
                        case PyType::MODULE:
                            throw std::invalid_argument("Not implemented yet");
                            break;
                        case PyType::SPATIAL_LATTICE:
                            param.addParameter<SpatialLattice<float>>(key, pyvalue.cast<SpatialLattice<float>>());
                            break;
                        case PyType::UNKNOWN:
                            py::str type_str = py::str(pyvalue.get_type().attr("__name__"));
                            std::string type_name = type_str.operator std::string();
                            throw std::invalid_argument("Unsupported type: " + type_name);
                            break;
                    }
                }
                value = param;
                return true;
            } else {
                return false;
            }

        }

        // Conversion from C++ to Python
        static handle cast(const NSL::Parameter& src, return_value_policy, handle) {
            // Implement conversion logic here
            return py::cast(src).release();
        }
    };

} // namespace detail
} // namespace pybind11
