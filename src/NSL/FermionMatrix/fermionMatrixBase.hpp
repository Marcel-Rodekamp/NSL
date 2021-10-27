#ifndef NSL_FERM_BASE_HPP
#define NSL_FERM_BASE_HPP


#include<vector>
#include "../Lattice/lattice.hpp"
#include "../assert.hpp"
#include "../complex.hpp"

#include "../Tensor/tensor.hpp"



template<typename Type>


//definition of class FermionMatrixBase

class FermionMatrixBase: 
{

public:

NSL::TimeTensor<Type> M(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
NSL::TimeTensor<Type> M_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
NSL::TimeTensor<Type> MM_dagger(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> & psi);
Type det_M;






//constructor

FermionMatrixBase( NSL::Lattice::SpatialLattice & L)
:Lat(&L)
{
}



private:

NSL::Lattice::SpatialLattice * Lat;

};

#endif