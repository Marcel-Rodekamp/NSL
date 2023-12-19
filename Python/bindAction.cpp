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

    // template <template <typename...> class ActionType, NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    // void bindActionImplementation(py::module &m, std::string class_name){
    //     using TensorT = Tensor<TensorType>;
    //     py::class_<ActionType, NSL::Action::BaseAction<Type, TensorType>>(m, class_name.c_str())
    //         .def(py::init<NSL::Parameter &>())
    //         .def(py::init<NSL::Parameter &, const std::string &>());
    //         // .def("eval", py::overload_cast<const TensorT&>(&ActionType::eval))
    //         // .def("grad", py::overload_cast<const TensorT&>(&ActionType::grad))
    //         // .def("force", py::overload_cast<const TensorT&>(&ActionType::force));
    // }

    // template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    // void bindHubbardGaugeAction(py::module &m, std::string class_name){
    //     NSL::Action::HubbardGaugeAction<Type, TensorType> act(NSL::Parameter());
    // }
    // template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    // void bindHubbardFermionAction(py::module &m, std::string class_name){
    //     using LatticeType = NSL::Lattice::SpatialLattice<Type>;
    //     using FermionMatrixType = NSL::FermionMatrix::HubbardExp<Type, LatticeType>;
    //     NSL::Action::HubbardFermionAction<Type, LatticeType, FermionMatrixType, TensorType> act(NSL::Parameter());
    //     // using TensorT = Tensor<float>;
    //     // py::class_<NSL::Action::HubbardFermionAction<float, float>>(m, class_name.c_str())
    //     //     .def(py::init<NSL::Parameter &>())
    //     //     .def(py::init<NSL::Parameter &, const std::string &>());
    //         // .def("eval", py::overload_cast<const TensorT&>(&ActionType::eval))
    //         // .def("grad", py::overload_cast<const TensorT&>(&ActionType::grad))
    //         // .def("force", py::overload_cast<const TensorT&>(&ActionType::force));
    // }

    // Base class
    class Base {
    public:
        virtual ~Base() = default;
        virtual void foo() = 0;
    };

    // Derived class template
    template <typename T>
    class Derived : public Base {
    public:
        void foo() override {
            std::cout << typeid(T).name() << std::endl;
        }
    };

    // Factory function
    std::shared_ptr<Base> createDerived(int type) {
        if (type == 1) {
            return std::make_shared<Derived<int>>();
        } else if (type == 2) {
            return std::make_shared<Derived<float>>();
        }
        // Add more cases as needed...
        return nullptr;
    }

    std::shared_ptr<NSL::Action::BaseAction<float, float>> createHubbardGaugeAction(NSL::Parameter & params) {
        return std::make_shared<NSL::Action::HubbardGaugeAction<float, float>>(params);
    }

    std::shared_ptr<NSL::Action::BaseAction<complex<double>, complex<double>>> createHubbardFermionAction(NSL::Parameter & params) {
        using LatticeType = NSL::Lattice::SpatialLattice<complex<double>>;
        using FermionMatrixType = NSL::FermionMatrix::HubbardExp<complex<double>, LatticeType>;
        return std::make_shared<NSL::Action::HubbardFermionAction<complex<double>, LatticeType, FermionMatrixType, complex<double>>>(params);
    }

    void bindAction(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_action = m.def_submodule("Action");
        
        bindBaseAction<complex<double>, complex<double>>(m_action, "BaseAction_cc");
        bindBaseAction<float, float>(m_action, "BaseAction_ff");
        bindBaseAction<double, double>(m_action, "BaseAction_dd");
        // bindHubbardFermionAction<float, float>(m_action, "HubbardFermionAction");
        // bindActionImplementation<NSL::Action::HubbardGaugeAction, float, float>(m_action, "HubbardGaugeAction");
        // using LatticeType = NSL::Lattice::SpatialLattice<float>;
        // using FermionMatrixType = NSL::FermionMatrix::FermionMatrix<float, LatticeType>;
        // bindActionImplementation<NSL::Action::HubbardFermionAction, float, float>(m_action, "HubbardGaugeAction");
        py::class_<Base, std::shared_ptr<Base>>(m_action, "Base")
            .def("foo", &Base::foo);

        m_action.def("createDerived", &createDerived);
        m_action.def("HubbardGaugeAction", &createHubbardGaugeAction);
        m_action.def("HubbardFermionAction", &createHubbardFermionAction);
        // Set default aliases
        m_action.attr("BaseAction") = m_action.attr("BaseAction_cc");
    }
}
