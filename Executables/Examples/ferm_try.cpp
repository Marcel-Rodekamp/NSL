
//#include <complex>
#include "complex.hpp"
//#include "../test.hpp"
#include <typeinfo>
#include "Lattice/lattice.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixBase.hpp"

int main()

{float size = 6, kappa = 1;

NSL::Lattice::Ring<float> ring(size, kappa);
NSL::Lattice::SpatialLattice<float>* p = &ring; 
NSL::FermionMatrix::FermionMatrixBase<float> base(p);

return(0);

}

