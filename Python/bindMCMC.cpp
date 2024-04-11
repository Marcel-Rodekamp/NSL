#include "../src/NSL/MCMC.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL::MCMC;

namespace NSL::Python {
    
    template<NSL::Concept::isNumber Type>
    void bindMarkovState(py::module &m, const std::string &name) {
        py::class_<MarkovState<Type>>(m, name.c_str())
            .def(py::init<NSL::Configuration<Type>, const Type &, const NSL::RealTypeOf<Type> &, NSL::size_t, bool>(),
                "config"_a, "actionValue"_a, "acceptanceProbability"_a, "markovTime"_a, "accepted"_a)
            .def(py::init<NSL::Configuration<Type>, const Type &, const NSL::RealTypeOf<Type> &>(),
                "config"_a, "actionValue"_a, "acceptanceProbability"_a)
            .def_readwrite("configuration", &MarkovState<Type>::configuration)
            .def_readwrite("actionValue", &MarkovState<Type>::actionValue)
            .def_readwrite("weights", &MarkovState<Type>::weights)
            .def_readwrite("acceptanceProbability", &MarkovState<Type>::acceptanceProbability)
            .def_readwrite("markovTime", &MarkovState<Type>::markovTime)
            .def_readwrite("accepted", &MarkovState<Type>::accepted);
    }

    void bindMCMC(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_mcmc = m.def_submodule("MCMC");
        
        bindMarkovState<NSL::complex<double>>(m_mcmc, "MarkovState");
        
        
    }
}