#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

enum class PyGenType {
    BOOL,
    INT,
    SIZE_T,
    FLOAT,
    DOUBLE,
    COMPLEX_FLOAT,
    COMPLEX_DOUBLE,
    STR,
    DEVICE,
    UNKNOWN
};

PyGenType get_pytype(py::handle& pyvalue) {
    if (py::isinstance<py::bool_>(pyvalue)) {
        return PyGenType::BOOL;
    } else if (py::isinstance<py::int_>(pyvalue)) {
        return PyGenType::INT;
    } else if (py::isinstance<py::ssize_t>(pyvalue)) {
        return PyGenType::SIZE_T;
    } else if (py::isinstance<py::float_>(pyvalue)) {
        return PyGenType::FLOAT;
    } else if (py::isinstance<py::array_t<double>>(pyvalue)) {
        return PyGenType::DOUBLE;
    } else if (py::type::of(pyvalue).is(py::module::import("builtins").attr("complex"))) {
         return PyGenType::COMPLEX_DOUBLE;
    } else if (py::isinstance<py::array_t<std::complex<double>>>(pyvalue)) {
        return PyGenType::COMPLEX_DOUBLE;
    } else if (py::isinstance<py::str>(pyvalue)) {
        return PyGenType::STR;
    } else if (py::isinstance<py::capsule>(pyvalue)) {
        return PyGenType::DEVICE;
    } else {
        std::cout << py::type::of(pyvalue) << std::endl;
        return PyGenType::UNKNOWN;
    }
}
// PYBIND11_MAKE_OPAQUE(NSL::GenType);
// PYBIND11_MAKE_OPAQUE(std::variant<bool,int, NSL::size_t, float, double, NSL::complex<float>, NSL::complex<double>, std::string, NSL::Device>);
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
                    switch (get_pytype(item.second)) {
                        case PyGenType::BOOL:
                            param[key] = item.second.cast<bool>();
                            break;
                        case PyGenType::INT:
                            param[key] = item.second.cast<int>();
                            break;
                        case PyGenType::SIZE_T:
                            param[key] = item.second.cast<NSL::size_t>();
                            break;
                        case PyGenType::FLOAT:
                            param[key] = item.second.cast<float>();
                            break;
                        case PyGenType::DOUBLE:
                            param[key] = item.second.cast<double>();
                            break;
                        case PyGenType::COMPLEX_FLOAT:
                            param[key] = item.second.cast<NSL::complex<float>>();
                            break;
                        case PyGenType::COMPLEX_DOUBLE:
                            param[key] = item.second.cast<NSL::complex<double>>();
                            break;
                        case PyGenType::STR:
                            param[key] = item.second.cast<std::string>();
                            break;
                        case PyGenType::DEVICE:
                            param[key] = item.second.cast<NSL::Device>();
                            break;
                        default:
                            throw std::runtime_error("Unsupported Python type");
                            break;
                    }
                    // NSL::GenType pyvalue = item.second.cast< std::variant<bool,int, NSL::size_t, float, double, NSL::complex<float>, NSL::complex<double>, std::string, NSL::Device>>();
                    // std::cout << key << " " << pyvalue << " " << item.second.get_type() << std::endl;

                    // param[key] = pyvalue;
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
            std::cout << "Casting NSL::Parameter to Python" << std::endl;
            return py::cast(src).release();
        }
    };

} // namespace detail
} // namespace pybind11
