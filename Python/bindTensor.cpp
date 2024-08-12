#include <torch/torch.h>
#include <torch/extension.h> 

namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL;

namespace NSL::Python { 
    template <NSL::Concept::isNumber Type>
    void printTensor(Tensor<Type> &tensor){
        std::cout << tensor << std::endl;
    }

    template <NSL::Concept::isNumber Type>
    void bindTensor(py::module &m, std::string class_name){
        auto tensor_class = py::class_<Tensor<Type>>(m, class_name.c_str())
            .def(py::init<>())
            .def(py::init<const torch::Tensor &>())   
            .def("to_torch", [](Tensor<Type> &self){
                return torch::Tensor(self);
            });

        m.def("printTensor", [](Tensor<Type> &x){
                printTensor(x);
            });
    }

    void bindTensor(py::module &m) {
        // ToDo: Documentation
        // bindTensor<short>(m, "Tensor_short");
        bindTensor<int>(m, "Tensor_int");
        // // bindTensor<long int>(m, "Tensor<long int>");         not possible
        bindTensor<float>(m, "Tensor_float");
        bindTensor<double>(m, "Tensor_double");
        // bindTensor<long double>(m, "Tensor_long_double");
        // bindTensor<complex<short>>(m, "Tensor_complex_short");
        // bindTensor<complex<int>>(m, "Tensor_complex_int");
        // bindTensor<complex<float>>(m, "Tensor_complex_float");
        // bindTensor<complex<double>>(m, "Tensor_complex_double");
        // bindTensor<complex<long double>>(m, "Tensor_complex_long_double");

        
    }
}

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define TYPE_NAME(Type) "NSL::Tensor<" TOSTRING(Type) ">"

namespace pybind11 {
namespace detail {
    template <typename Type> 
    struct type_caster<NSL::Tensor<Type>> {
    public:
        PYBIND11_TYPE_CASTER(NSL::Tensor<Type>, _(TYPE_NAME(Type)));

        // Conversion from Python to C++
        bool load(handle src, bool) {
            /* Extract the PyTorch tensor from the Python object */
            // print type of src
            if(py::str(src.get_type().attr("__name__")).operator std::string() == "Tensor"){
                torch::Tensor src_tensor = src.cast<torch::Tensor>();
                // std::cout << src_tensor << std::endl;
                /* Convert the PyTorch tensor to NSL::Tensor<Type> */
                value = NSL::Tensor<Type>(src_tensor);
            }else{
                return false;
            }
            return true;
        }

        // Conversion from C++ to Python
        static handle cast(const NSL::Tensor<Type>& src, return_value_policy, handle) {
            /* Convert the NSL::Tensor<Type> to PyTorch tensor */
            torch::Tensor dst_tensor = torch::Tensor(src);

            /* Convert the PyTorch tensor to a Python object and return it */
            return py::cast(dst_tensor).release();
        }
    };
} // namespace detail
} // namespace pybind11

// template struct pybind11::detail::type_caster<NSL::Tensor<short>>;
template struct pybind11::detail::type_caster<NSL::Tensor<int>>;
template struct pybind11::detail::type_caster<NSL::Tensor<float>>;
template struct pybind11::detail::type_caster<NSL::Tensor<double>>;
// template struct pybind11::detail::type_caster<NSL::Tensor<long double>>;
// template struct pybind11::detail::type_caster<NSL::Tensor<complex<float>>>;
template struct pybind11::detail::type_caster<NSL::Tensor<complex<double>>>;
// template struct pybind11::detail::type_caster<NSL::Tensor<complex<long double>>>;

#undef STRINGIFY
#undef TOSTRING
#undef TYPE_NAME