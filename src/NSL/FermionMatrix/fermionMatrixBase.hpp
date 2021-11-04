#ifndef NSL_FERM_BASE_HPP
#define NSL_FERM_BASE_HPP


#include<vector>
#include "../Lattice/lattice.hpp"
#include "../assert.hpp"
#include "../complex.hpp"

#include "../Tensor/tensor.hpp"

namespace NSL::FermionMatrix{

//definition of class FermionMatrixBase

template<typename Type>
class FermionMatrixBase {

public:

NSL::TimeTensor<Type> M(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi, NSL::Tensor<Type> & expKappa);
NSL::TimeTensor<Type> M_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
NSL::TimeTensor<Type> MM_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
Type det_M;






//constructor
FermionMatrixBase()
{}
FermionMatrixBase( NSL::Lattice::SpatialLattice<Type>* L); 




private:

NSL::Lattice::SpatialLattice<Type>* Lat;

};

}

#endif