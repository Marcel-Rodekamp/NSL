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
        bindTensor<short>(m, "Tensor_short");
        bindTensor<int>(m, "Tensor_int");
        // // bindTensor<long int>(m, "Tensor<long int>");         not possible
        bindTensor<float>(m, "Tensor_float");
        bindTensor<double>(m, "Tensor_double");
        bindTensor<long double>(m, "Tensor_long_double");
        bindTensor<complex<short>>(m, "Tensor_complex_short");
        bindTensor<complex<int>>(m, "Tensor_complex_int");
        bindTensor<complex<float>>(m, "Tensor_complex_float");
        bindTensor<complex<double>>(m, "Tensor_complex_double");
        bindTensor<complex<long double>>(m, "Tensor_complex_long_double");

        
    }
}