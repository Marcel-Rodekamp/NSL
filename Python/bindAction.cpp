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
        py::class_<NSL::Action::BaseAction<Type, TensorType>, std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>>>(m, class_name.c_str())
            .def("eval", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("eval", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("grad", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("grad", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("force", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force))
            .def("force", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force));
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> createHubbardGaugeAction(NSL::Parameter & params) {
        std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> hga = std::make_unique<NSL::Action::HubbardGaugeAction<Type, TensorType>>(params);
        return hga;
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> createHubbardFermionAction(NSL::Lattice::SpatialLattice<Type> & lattice, NSL::Parameter & params) {
        using FermionMatrixType = NSL::FermionMatrix::HubbardExp<Type, NSL::Lattice::SpatialLattice<Type>>;
        std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> hfa = std::make_unique<NSL::Action::HubbardFermionAction<Type, NSL::Lattice::SpatialLattice<Type>, FermionMatrixType, TensorType>>(lattice, params);
        return hfa;
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> createSumAction(std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> action1, std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> action2) {
        std::shared_ptr<NSL::Action::BaseAction<Type, TensorType>> sumAction = std::make_shared<NSL::Action::Action<NSL::Action::BaseAction<Type, TensorType>, NSL::Action::BaseAction<Type, TensorType>>>(action1, action2);
        return sumAction;
    }

    template <typename Type, typename TensorType>
    void bindActionWithTypes(py::module &m_action, const std::string &nameAppendix) {
        bindBaseAction<Type, TensorType>(m_action, "BaseAction_" + nameAppendix);
        bindActionImplementations<Type, TensorType>(m_action, nameAppendix);
    }

    template <typename Type, typename TensorType>
    void bindActionImplementations(py::module &m_action, const std::string &nameAppendix) {
        m_action.def(("HubbardGaugeAction_" + nameAppendix).c_str(), &createHubbardGaugeAction<Type, TensorType>);
        m_action.def(("HubbardFermionAction_" + nameAppendix).c_str(), &createHubbardFermionAction<Type, TensorType>);
    }
    
    class SumAction {
        py::args _baseActionList;
        public:
        SumAction(py::args BaseActions) {
            _baseActionList = BaseActions;
        }

        NSL::complex<double> eval(const NSL::Tensor<NSL::complex<double>> & tensor) {
            NSL::complex<double> sum = 0.0;
            for (auto baseAction : _baseActionList) {
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> eval(tensor);
            }
            return sum;
        }

        NSL::Configuration<NSL::complex<double>> grad(const NSL::Tensor<NSL::complex<double>> & tensor) {
            NSL::Configuration<NSL::complex<double>> sum;
            for (auto baseAction : _baseActionList) {
                // std::cout << baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> grad(tensor) << std::endl;
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> grad(tensor);
            }
            return sum;
        }

        NSL::Configuration<NSL::complex<double>> force(const NSL::Tensor<NSL::complex<double>> & tensor) {
            NSL::Configuration<NSL::complex<double>> sum;
            for (auto baseAction : _baseActionList) {
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> force(tensor);
            }
            return sum;
        }

    };
    void bindAction(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        // ToDo: Configuration dicts as input
        py::module m_action = m.def_submodule("Action");
        
        bindActionWithTypes<float, float>(m_action, "ff");
        bindActionWithTypes<complex<double>, complex<double>>(m_action, "cc");
        
        py::class_<SumAction>(m_action, "SumAction")
            .def(py::init<py::args>())
            .def("eval", py::overload_cast<const Tensor<NSL::complex<double>>&>(&SumAction::eval))
            // .def("eval", py::overload_cast<Configuration<NSL::complex<double>>&>(&SumAction::eval))
            .def("grad", [](SumAction &self, const Tensor<NSL::complex<double>> & tensor) {
                return self.grad(tensor);
            })
            // .def("grad", py::overload_cast<Configuration<NSL::complex<double>>&>(&SumAction::grad))
            .def("force", [](SumAction &self, const Tensor<NSL::complex<double>> & tensor) {
                return self.force(tensor);
            });
            // .def("force", py::overload_cast<Configuration<NSL::complex<double>>&>(&SumAction::force));

        
        // Set default aliases
        m_action.attr("BaseAction") = m_action.attr("BaseAction_cc");
        m_action.attr("HubbardGaugeAction") = m_action.attr("HubbardGaugeAction_cc");
        m_action.attr("HubbardFermionAction") = m_action.attr("HubbardFermionAction_cc");
    }
}
