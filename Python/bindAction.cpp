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
        py::class_<NSL::Action::BaseAction<Type, TensorType>, std::unique_ptr<NSL::Action::BaseAction<Type, TensorType>>>(m, class_name.c_str())
            .def("eval", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("eval", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::eval))
            .def("grad", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("grad", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::grad))
            .def("force", py::overload_cast<const Tensor<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force))
            .def("force", py::overload_cast<Configuration<TensorType>&>(&NSL::Action::BaseAction<Type, TensorType>::force));
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    std::unique_ptr<NSL::Action::BaseAction<Type, TensorType>> createHubbardGaugeAction(NSL::Parameter & params) {
        std::unique_ptr<NSL::Action::BaseAction<Type, TensorType>> hga = std::make_unique<NSL::Action::HubbardGaugeAction<Type, TensorType>>(params);
        return hga;
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    std::unique_ptr<NSL::Action::BaseAction<Type, TensorType>> createHubbardFermionAction(NSL::Lattice::SpatialLattice<Type> & lattice, NSL::Parameter & params) {
        using FermionMatrixType = NSL::FermionMatrix::HubbardExp<Type, NSL::Lattice::SpatialLattice<Type>>;
        std::unique_ptr<NSL::Action::BaseAction<Type, TensorType>> hfa = std::make_unique<NSL::Action::HubbardFermionAction<Type, NSL::Lattice::SpatialLattice<Type>, FermionMatrixType, TensorType>>(lattice, params);
        return hfa;
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
    
    class SumAction : public NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>{
        public:
        SumAction(py::args BaseActions) : NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>("phi"){
            _baseActionList = BaseActions;
        }

        NSL::complex<double> eval(Configuration<NSL::complex<double>> & config) {
            NSL::complex<double> sum = 0.0;
            for (auto baseAction : _baseActionList) {
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> eval(config);
            }
            return sum;
        }

        NSL::complex<double> eval(const Tensor<NSL::complex<double>> & tensor) {
            throw std::runtime_error("This object is a SumAction, it can only be evaluated on a Configuration object");
        }

        NSL::Configuration<NSL::complex<double>> grad(Configuration<NSL::complex<double>> & config) {
            NSL::Configuration<NSL::complex<double>> sum;
            for (auto baseAction : _baseActionList) {
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> grad(config);
            }
            return sum;
        }

        NSL::Configuration<NSL::complex<double>> grad(const Tensor<NSL::complex<double>> & tensor) {
            throw std::runtime_error("This object is a SumAction, it can only be evaluated on a Configuration object");
        }

        NSL::Configuration<NSL::complex<double>> force(Configuration<NSL::complex<double>> & config) {
            NSL::Configuration<NSL::complex<double>> sum;
            for (auto baseAction : _baseActionList) {
                sum += baseAction.cast<NSL::Action::BaseAction<NSL::complex<double>, NSL::complex<double>>*>() -> force(config);
            }
            return sum;
        }

        NSL::Configuration<NSL::complex<double>> force(const Tensor<NSL::complex<double>> & tensor) {
            throw std::runtime_error("This object is a SumAction, it can only be evaluated on a Configuration object");
        }

        NSL::complex<double> operator()(Configuration<NSL::complex<double>> & config) {
            return eval(config);
        }

        private:
        py::args _baseActionList;
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
            // .def("eval", py::overload_cast<const Tensor<NSL::complex<double>>&>(&SumAction::eval))
            .def("eval", py::overload_cast<NSL::Configuration<NSL::complex<double>>&>(&SumAction::eval))
            .def("grad", [](SumAction &self, NSL::Configuration<NSL::complex<double>> & config) {
                return self.grad(config);
            })
            // // .def("grad", py::overload_cast<Configuration<NSL::complex<double>>&>(&SumAction::grad))
            .def("force", [](SumAction &self, NSL::Configuration<NSL::complex<double>> & config) {
                return self.force(config);
            })
            // .def("force", py::overload_cast<Configuration<NSL::complex<double>>&>(&SumAction::force));
            .def("__call__", [](SumAction &self, NSL::Configuration<NSL::complex<double>> & config) {
                return self(config);
            });

        
        // Set default aliases
        m_action.attr("BaseAction") = m_action.attr("BaseAction_cc");
        m_action.attr("HubbardGaugeAction") = m_action.attr("HubbardGaugeAction_cc");
        m_action.attr("HubbardFermionAction") = m_action.attr("HubbardFermionAction_cc");
    }
}
