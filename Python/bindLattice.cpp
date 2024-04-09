#include "../src/NSL/Lattice/lattice.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL::Lattice;

namespace NSL::Python {
    template <typename Type>
    void bindSpatialLattice(py::module &m, std::string class_name){
        py::class_<SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const std::string &, const NSL::Tensor<Type> &, const NSL::Tensor<double> &>())
            .def("__call__", &SpatialLattice<Type>::operator())
            .def("__repr__", &SpatialLattice<Type>::name)
            .def("coordinates", &SpatialLattice<Type>::coordinates)
            .def("sites", &SpatialLattice<Type>::sites)
            .def("name", &SpatialLattice<Type>::name)
            .def("adjacency_matrix", &SpatialLattice<Type>::adjacency_matrix)
            .def("hopping_matrix", &SpatialLattice<Type>::hopping_matrix, "delta"_a = 1.)
            .def("exp_hopping_matrix", &SpatialLattice<Type>::exp_hopping_matrix, "delta"_a = 1.)
            .def("eigh_hopping", &SpatialLattice<Type>::eigh_hopping, "delta"_a = 1.)
            .def("bipartite", &SpatialLattice<Type>::bipartite)
            .def("to", [](SpatialLattice<Type>& self, const std::string& device_identifier, const NSL::size_t ID) {     //TODO bind NSL::Device for better style
                NSL::Device device = NSL::Device(device_identifier, ID);
                self.to(device);
            }, "device"_a = "CPU", "ID"_a = 0)
            .def("device", &SpatialLattice<Type>::device);
    }

    template <typename Type>
    void bindComplete(py::module &m, std::string class_name){
        py::class_<Complete<Type>, SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const std::size_t, const Type &, const double &>(), "n"_a, "kappa"_a = 1., "radius"_a = 1.)
            .def("bipartite", &Complete<Type>::bipartite);
    }

    template <typename Type>
    void bindTriangle(py::module &m, std::string class_name){
        py::class_<Triangle<Type>, Complete<Type>>(m, class_name.c_str())
            .def(py::init<const Type &, const double &>(), "kappa"_a = 1., "radius"_a = 1.)
            .def("bipartite", &Triangle<Type>::bipartite);
    }

    template <typename Type>
    void bindTetrahedon(py::module &m, std::string class_name){
        py::class_<Tetrahedron<Type>, Complete<Type>>(m, class_name.c_str())
            .def(py::init<const Type &, const double &>(), "kappa"_a = 1., "edge"_a = 1.)
            .def("bipartite", &Tetrahedron<Type>::bipartite);
    }

    template <typename Type>
    void bindGeneric(py::module &m, std::string class_name){
        py::class_<Generic<Type>, SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const YAML::Node &, const Type &>(), "system"_a, "kappa"_a = 1.);
    }

    template <typename Type>
    void bindHoneycomb(py::module &m, std::string class_name){
        py::class_<Honeycomb<Type>, SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const std::vector<int>, const Type &>());
    }

    template <typename Type>
    void bindRing(py::module &m, std::string class_name){
        py::class_<Ring<Type>, SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const std::size_t, const Type &, const double &>());
    }

    template <typename Type>
    void bindSquare(py::module &m, std::string class_name){
        py::class_<Square<Type>, SpatialLattice<Type>>(m, class_name.c_str())
            .def(py::init<const std::vector<std::size_t>, const std::vector<Type> &, const std::vector<double>>())
            .def(py::init<const std::vector<std::size_t>, const std::vector<Type> &, const double>())
            .def(py::init<const std::vector<std::size_t>, const Type &, const double>())
            .def(py::init<const std::vector<std::size_t>, const Type &, const std::vector<double>>());
    }

    template <typename Type>
    void bindCube3D(py::module &m, std::string class_name){
        py::class_<Cube3D<Type>, Square<Type>>(m, class_name.c_str())
            .def(py::init<std::size_t, const Type &, const double &>());
    }



    void bindLattice(py::module &m) {
        // ToDo: Templating
        // ToDo: Documentation
        py::module m_lattice = m.def_submodule("Lattice");
        
        bindSpatialLattice<NSL::complex<double>>(m_lattice, "SpatialLattice");
        bindComplete<NSL::complex<double>>(m_lattice, "Complete");
        bindTriangle<NSL::complex<double>>(m_lattice, "Triangle");
        bindTetrahedon<NSL::complex<double>>(m_lattice, "Tetrahedron");
        bindGeneric<NSL::complex<double>>(m_lattice, "Generic");
        bindHoneycomb<NSL::complex<double>>(m_lattice, "Honeycomb");
        bindRing<NSL::complex<double>>(m_lattice, "Ring");
        bindSquare<NSL::complex<double>>(m_lattice, "Square");
        bindCube3D<NSL::complex<double>>(m_lattice, "Cube3D");
    }
}

namespace pybind11 {
    namespace detail {
        template <>
        struct type_caster<YAML::Node> {
        public:
            PYBIND11_TYPE_CASTER(YAML::Node, _("YAML::Node"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                std::string yaml_path = src.cast<std::string>();
                value = YAML::LoadFile(yaml_path);
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const YAML::Node& src, return_value_policy /* policy */, handle /* parent */) {
                std::stringstream ss;
                ss << src;
                return py::str(ss.str()).release();
            }
        };
    }  // namespace detail
}  // namespace pybind11