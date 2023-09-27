#include "NSL.hpp"
#include <pybind11/pybind11.h>
#include "bindLattice.cpp"
#include "bindTensor.cpp"
#include "bindParameter.cpp"


namespace NSL::Python {
    PYBIND11_MODULE(PyNSL, m) {
        bindLattice(m);
        bindTensor(m);
        bindParameter(m);
    }
}