#include "../src/NSL/Action.hpp"
#include "../src/NSL/Lattice.hpp"
#include "../src/NSL/FermionMatrix.hpp"
#include "../src/NSL/concepts.hpp"
#include "Action/action.tpp"
#include <torch/torch.h>
#include <torch/extension.h> 

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

// using namespace NSL::Action;         // when I use this namespace I get a huge error message from torch

namespace NSL::Python {
    template <Concept::isNumber Type, Concept::isNumber TensorType>
    void bindBaseAction(py::module &m, std::string class_name){
        py::class_<Action::BaseAction<Type, TensorType>, std::shared_ptr<Action::BaseAction<Type, TensorType>>>(m, class_name.c_str())
            .def("eval", py::overload_cast<const Tensor<TensorType>&>(&Action::BaseAction<Type, TensorType>::eval))
            .def("eval", py::overload_cast<Configuration<TensorType>&>(&Action::BaseAction<Type, TensorType>::eval))

            .def("grad", py::overload_cast<const Tensor<TensorType>&>(&Action::BaseAction<Type, TensorType>::grad))
            .def("grad", py::overload_cast<Configuration<TensorType>&>(&Action::BaseAction<Type, TensorType>::grad))
            
            .def("force", py::overload_cast<const Tensor<TensorType>&>(&Action::BaseAction<Type, TensorType>::force))
            .def("force", py::overload_cast<Configuration<TensorType>&>(&Action::BaseAction<Type, TensorType>::force))

            ;
    }

    std::shared_ptr<Action::BaseAction<complex<double>, complex<double>>> createHubbardGaugeAction(Parameter & params) {
        return std::make_shared<Action::HubbardGaugeAction<complex<double>, complex<double>>>(params);
    }

    std::shared_ptr<Action::BaseAction<complex<double>, complex<double>>> createHubbardFermionAction( 
        Lattice::Generic<complex<double>> &lattice, Parameter & params
    ) {
        using LatticeType = Lattice::Generic<complex<double>>;
        using FermionMatrixType = FermionMatrix::HubbardExp<complex<double>, LatticeType>;

        return std::make_shared<
            Action::HubbardFermionAction<
                complex<double>, 
                LatticeType, 
                FermionMatrixType
            >
        >(lattice,params);
    }

    template<typename Type, class SingleAction1, class SingleAction2>
    void bindActionContainer(py::module &m, std::string class_name){

        py::class_<Action::Action<SingleAction1,SingleAction2>, std::shared_ptr<Action::Action<SingleAction1,SingleAction2>>>(m, class_name.c_str())
            .def(py::init( 
                [] (NSL::Action::BaseAction<Type,Type> * a1, NSL::Action::BaseAction<Type,Type> * a2) {
                     return std::make_shared<Action::Action<SingleAction1,SingleAction2>> ( 
                        *dynamic_cast<SingleAction1*>(a1),
                        *dynamic_cast<SingleAction2*>(a2)
                    );
                })
            )
            .def("eval", py::overload_cast<Configuration<Type>&>(&Action::Action<SingleAction1,SingleAction2>::template eval<Type>))
            .def("__call__", py::overload_cast<Configuration<Type>&>(&Action::Action<SingleAction1,SingleAction2>::template eval<Type>))
            .def("grad", py::overload_cast<Configuration<Type>&>(&Action::Action<SingleAction1,SingleAction2>::template grad<Type>))
            .def("force",py::overload_cast<Configuration<Type>&>(&Action::Action<SingleAction1,SingleAction2>::template force<Type>))
            // todo bind all the + operators...
            ;
    }

    void bindAction(py::module &m) {
        py::module m_action = m.def_submodule("Action");
        
        bindBaseAction<complex<double>, complex<double>>(m_action, "BaseAction_cc");
        bindBaseAction<float, float>(m_action, "BaseAction_ff");
        bindBaseAction<double, double>(m_action, "BaseAction_dd");

        m_action.def("HubbardGaugeAction", &createHubbardGaugeAction);
        m_action.def("HubbardFermionAction", &createHubbardFermionAction);
        // Set default aliases
        m_action.attr("BaseAction") = m_action.attr("BaseAction_cc");

        bindActionContainer<complex<double>,
            Action::HubbardGaugeAction<complex<double>, complex<double>>,
            Action::HubbardFermionAction<
                complex<double>, 
                Lattice::Generic<complex<double>>, 
                FermionMatrix::HubbardExp<complex<double>, Lattice::Generic<complex<double>>>
            >
        >(m_action, "HubbardAction_EXP_GEN");
    }
}
