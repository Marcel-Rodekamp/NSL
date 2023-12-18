#include "../src/NSL/Action/action.tpp"
#include <torch/torch.h>
#include <torch/extension.h> 

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

// using namespace NSL::Action;         // when I use this namespace I get a huge error message from torch

namespace NSL::Python {
    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    void bindBaseAction(py::module &m, std::string class_name){
        py::class_<NSL::Action::BaseAction<Type, TensorType>>(m, class_name.c_str())
            .def("eval", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("eval", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("grad", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("grad", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("force", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force))
            .def("force", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force));
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    void bindHubbardGaugeAction(py::module &m, std::string class_name){
        py::class_<NSL::Action::HubbardGaugeAction<Type, TensorType>, NSL::Action::BaseAction<Type, TensorType>>(m, class_name.c_str())
            .def(py::init<NSL::Parameter &>())
            .def(py::init<NSL::Parameter &, const std::string &>())
            .def("eval", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::HubbardGaugeAction<Type, TensorType>::eval))
            .def("grad", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::HubbardGaugeAction<Type, TensorType>::grad))
            .def("force", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::HubbardGaugeAction<Type, TensorType>::force));
    }

    void bindAction(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_action = m.def_submodule("Action");
        
        bindBaseAction<complex<double>, complex<double>>(m_action, "BaseAction_cc");
        bindBaseAction<float, float>(m_action, "BaseAction_ff");
        bindBaseAction<double, double>(m_action, "BaseAction_dd");
        bindHubbardGaugeAction<float, float>(m_action, "HubbardGaugeAction");

        // Set default aliases
        m_action.attr("BaseAction") = m_action.attr("BaseAction_cc");
    }
}