
#include "fermionMatrixBase.hpp"
#include "../Tensor/tensor.hpp"
#include "../Lattice/lattice.hpp"
//#include "../LinAlg/mat_vec.hpp"

namespace NSL::FermionMatrix {

template<typename Type>
NSL::FermionMatrix::FermionMatrixBase<Type>::FermionMatrixBase(NSL::Lattice::SpatialLattice<Type>* L)
:Lat(L)
{
}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixBase<Type>::M(NSL::TimeTensor<Type> & phi, NSL::TimeTensor<Type> &psi, NSL::Tensor<Type> & expKappa)
{

const long Nt = phi.shape(0);
const long Nx = phi.shape(1);

NSL::TimeTensor<Type> Mat(Nt,Nx,Nx), out(psi.shape());

c10::complex<double> num =(0,1);

for(std::size_t t=00; t<phi.shape(0); t++) {

//Mat(t)= (((phi(t)*num).exp().expand(phi.shape(1))).prod(expKappa));

}
Mat(0) *=-1;

//    for(std::size_t t = 0; t < psi.shape(0); ++t){
//        out(t) = NSL::LinAlg::mat_vec(Mat(t),(NSL::LinAlg::shift(psi,1))(t));
//    }

    return psi; //- out;

}


}

template class NSL::FermionMatrix::FermionMatrixBase<float>;
template class NSL::FermionMatrix::FermionMatrixBase<double>;

