#include "../src/NSL/Lattice/lattice.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

using namespace NSL::Lattice;

namespace NSL::Python {
    void bindLattice(py::module &m) {
        // py::class_<SpatialLattice>(m, "Lattice")
        //     .def()
    }
}