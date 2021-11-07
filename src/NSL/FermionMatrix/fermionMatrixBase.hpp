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


//Declaration of methods methods M, M_dagger, MM_dagger and M

NSL::TimeTensor<Type> M(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi, NSL::Tensor<Type> & expKappa);
NSL::TimeTensor<Type> M_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
NSL::TimeTensor<Type> MM_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
Type det_M;






//constructor
FermionMatrixBase()
{}
FermionMatrixBase( NSL::Lattice::SpatialLattice<Type>* L); 




private:

//a pointer pointing at the object of SpatialLattice type

NSL::Lattice::SpatialLattice<Type>* Lat;

};

}

#endif