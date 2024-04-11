#include "../src/NSL/Integrator.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL::Integrator;

namespace NSL::Python {
    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    void bindIntegratorBase(py::module &m, std::string class_name){
        py::class_<IntegratorBase<Type, TensorType>>(m, class_name.c_str())
            // .def(py::init<const NSL::Action::BaseAction<Type, TensorType>&>())
            .def("__call__", [](IntegratorBase<Type, TensorType> &integrator, py::args args){
                if (args.size() == 1){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    return integrator(q);
                } else if (args.size() == 2){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    NSL::Configuration<TensorType> p = args[1].cast<NSL::Configuration<TensorType>>();
                    return integrator(q, p);
                } else {
                    throw std::runtime_error("Invalid number of arguments");
                }
            });
    }

    template <NSL::Concept::isNumber TensorType>
    void bindLeapfrog(py::module &m, std::string class_name){
        py::class_<NSL::Integrator::Leapfrog<NSL::Python::SumAction>>(m, class_name.c_str())
            .def(py::init<const NSL::Python::SumAction&, const double&, const NSL::size_t&, bool>(), "action"_a, "trajectoryLength"_a, "numberSteps"_a, "backward"_a = false)
            .def("__call__", [](NSL::Integrator::Leapfrog<NSL::Python::SumAction> &integrator, py::args args){
                if (args.size() == 1){
                    throw std::runtime_error("Invalid number of arguments");
                } else if (args.size() == 2){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    NSL::Configuration<TensorType> p = args[1].cast<NSL::Configuration<TensorType>>();
                    return integrator(q, p);
                } else {
                    throw std::runtime_error("Invalid number of arguments");
                }
            });
    }

    template <NSL::Concept::isNumber TensorType>
    void bindLeapfrogRealForce(py::module &m, std::string class_name){
        py::class_<NSL::Integrator::LeapfrogRealForce<NSL::Python::SumAction>>(m, class_name.c_str())
            .def(py::init<const NSL::Python::SumAction&, const double&, const NSL::size_t&, bool>(), "action"_a, "trajectoryLength"_a, "numberSteps"_a, "backward"_a = false)
            .def("__call__", [](NSL::Integrator::LeapfrogRealForce<NSL::Python::SumAction> &integrator, py::args args){
                if (args.size() == 1){
                    throw std::runtime_error("Invalid number of arguments");
                } else if (args.size() == 2){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    NSL::Configuration<TensorType> p = args[1].cast<NSL::Configuration<TensorType>>();
                    return integrator(q, p);
                } else {
                    throw std::runtime_error("Invalid number of arguments");
                }
            });
    }

    template <NSL::Concept::isNumber TensorType>
    void bindRungeKutta2(py::module &m, std::string class_name){
        py::class_<NSL::Integrator::RungeKutta2<NSL::Python::SumAction>>(m, class_name.c_str())
            .def(py::init<const NSL::Python::SumAction&, const double&, const NSL::size_t&, bool>(), "action"_a, "maxTime"_a, "numberSteps"_a, "conjugategrad"_a = false)
            .def("__call__", [](NSL::Integrator::RungeKutta2<NSL::Python::SumAction> &integrator, py::args args){
                if (args.size() == 1){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    return integrator(q);
                } else {
                    throw std::runtime_error("Invalid number of arguments");
                }
            });
    }

    template <NSL::Concept::isNumber TensorType>
    void bindRungeKutta4(py::module &m, std::string class_name){
        py::class_<NSL::Integrator::RungeKutta4<NSL::Python::SumAction>>(m, class_name.c_str())
            .def(py::init<const NSL::Python::SumAction&, const double&, const NSL::size_t&, bool>(), "action"_a, "maxTime"_a, "numberSteps"_a, "conjugategrad"_a = false)
            .def("__call__", [](NSL::Integrator::RungeKutta4<NSL::Python::SumAction> &integrator, py::args args){
                if (args.size() == 1){
                    NSL::Configuration<TensorType> q = args[0].cast<NSL::Configuration<TensorType>>();
                    return integrator(q);
                } else {
                    throw std::runtime_error("Invalid number of arguments");
                }
            });
    }

    template <NSL::Concept::isNumber TensorType>
    void bindIntegratorImplementations(py::module &m, std::string class_name){
        bindLeapfrog<TensorType>(m, "Leapfrog");
        bindLeapfrogRealForce<TensorType>(m, "LeapfrogRealForce");
        bindRungeKutta2<TensorType>(m, "RungeKutta2");
        bindRungeKutta4<TensorType>(m, "RungeKutta4");
    }

    template <NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
    void bindIntegrators(py::module &m, std::string class_name){
        bindIntegratorBase<Type, TensorType>(m, class_name);
        bindIntegratorImplementations<TensorType>(m, class_name);
    }


    void bindIntegrator(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_integrator = m.def_submodule("Integrator");
        
        bindIntegrators<NSL::complex<double>, NSL::complex<double>>(m_integrator, "IntegratorBase");
        
        
    }
}