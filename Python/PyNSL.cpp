#include "NSL.hpp"
#include <pybind11/pybind11.h>
#include "bindLattice.cpp"


namespace NSL::Python {
    PYBIND11_MODULE(PyNSL, m) {
        bindLattice(m);
    }
}