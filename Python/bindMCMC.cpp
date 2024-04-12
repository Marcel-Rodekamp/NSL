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

    template< NSL::Concept::isTemplateDerived<NSL::Integrator::Integrator> IntegratorType, typename ActionType, NSL::Concept::isNumber Type>
    void bindHMC(py::module &m, const std::string &name) {
        using HMCType = NSL::MCMC::HMC<IntegratorType, ActionType>;
        py::class_<HMCType>(m, name.c_str())
            .def(py::init<const IntegratorType &, const ActionType &, NSL::H5IO &>())
            .def("generate", [](HMCType& hmc, const NSL::MCMC::MarkovState<Type>& state) {
                return hmc.generate(state);
            })
            .def("generate", [](HMCType& hmc, NSL::Configuration<Type>& config) {
                return hmc.generate(config);
            })
            .def("generate", [](HMCType& hmc, const NSL::MCMC::MarkovState<Type>& state, NSL::size_t Nconf, NSL::size_t saveFrequency, std::string baseNode) {
                return hmc.template generate<Chain::AllStates, Type>(state, Nconf, saveFrequency, baseNode);
            }, "state"_a, "Nconf"_a, "saveFrequency"_a = 1, "baseNode"_a = "markovChain")
            .def("generate", [](NSL::MCMC::HMC<IntegratorType, ActionType>& hmc, NSL::Configuration<Type>& config, NSL::size_t Nconf, NSL::size_t saveFrequency) {
                return hmc.template generate<Chain::AllStates, Type>(config, Nconf, saveFrequency);
            }, "config"_a, "Nconf"_a, "saveFrequency"_a = 1)
            .def("thermalize", [](HMCType& hmc, const NSL::MCMC::MarkovState<Type>& state, NSL::size_t Nconf, NSL::size_t saveFrequency, std::string baseNode) {
                return hmc.template generate<Chain::LastState, Type>(state, Nconf, saveFrequency, baseNode);
            }, "state"_a, "Nconf"_a, "saveFrequency"_a = 1, "baseNode"_a = "markovChain")
            .def("thermalize", [](HMCType& hmc, NSL::Configuration<Type>& config, NSL::size_t Nconf, NSL::size_t saveFrequency) {
                return hmc.template generate<Chain::LastState, Type>(config, Nconf, saveFrequency);
            }, "config"_a, "Nconf"_a, "saveFrequency"_a = 1);
    }

    void bindMCMC(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_mcmc = m.def_submodule("MCMC");
        
        bindMarkovState<NSL::complex<double>>(m_mcmc, "MarkovState");
        bindHMC<NSL::Integrator::Leapfrog<NSL::Python::SumAction>, NSL::Python::SumAction, NSL::complex<double>>(m_mcmc, "HMC");
        
        
    }
}