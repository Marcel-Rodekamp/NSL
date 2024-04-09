#include "NSL.hpp"
#include <pybind11/pybind11.h>
#include "bindDevice.cpp"
#include "bindLattice.cpp"
#include "bindTensor.cpp"
#include "bindConfiguration.cpp"
#include "bindParameter.cpp"
#include "bindAction.cpp"


namespace NSL::Python {
    PYBIND11_MODULE(PyNSL, m) {
        bindLattice(m);
        bindTensor(m);
        bindAction(m);
    }
}