#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define TYPE_NAME_CONFIG(Type) "NSL::Configuration<" TOSTRING(Type) ">"

namespace pybind11 {
namespace detail {

    template <NSL::Concept::isNumber Type> 
    struct type_caster<NSL::Configuration<Type>> {
        PYBIND11_TYPE_CASTER(NSL::Configuration<Type>, _(TYPE_NAME_CONFIG(Type)));

        // Conversion from Python to C++
        bool load(handle src, bool) {
            if (!isinstance<dict>(src)) {
                return false;
            }
            value.clear();
            py::dict src_dict = src.cast<py::dict>();
            for (auto item : src_dict) {
                std::string key = item.first.cast<std::string>();
                NSL::Tensor<Type> val = item.second.cast<NSL::Tensor<Type>>();
                value[key] = val;
            }
            return true;
        }

        // Conversion from C++ to Python
        static handle cast(const Configuration<Type>& src, return_value_policy policy, handle parent) {
            py::dict out_dict;
            for (const auto& item : src) {
                out_dict[item.first.c_str()] = py::cast(item.second, policy, parent);
            }
            return out_dict.release();
        }
    };
} // namespace detail
} // namespace pybind11

template struct pybind11::detail::type_caster<NSL::Configuration<NSL::complex<double>>>;