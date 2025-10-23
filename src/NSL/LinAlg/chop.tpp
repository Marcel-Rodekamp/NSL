#ifndef NSL_LINALG_CHOP_TPP
#define NSL_LINALG_CHOP_TPP

namespace NSL::LinAlg{

//! Chops elements (ie sets to zero) of tensor with relative size (compared to diagonal) less than some tolerance
/*!
 * This function compares relative sizes of elements relative to diagonal and sets them to zero if below some specified tolerance 
 * If Type is complex a runtime error is thrown. 
 * */
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> chop(const NSL::Tensor<Type>& tensor, NSL::RealTypeOf<Type> tol) {
    NSL::Tensor<Type> tensorChop = tensor;
    NSL::Tensor<Type> diagonals = abs(NSL::LinAlg::diag(tensor)); // determine absolute values of the diagonals
    int nx = diagonals.shape()[0];

    for (int i=0; i < nx; i++) {
      if (diagonals[i].real() <= tol ) continue;
      for (int j=0; j < nx; j++) {
      	  if (i==j) continue;
	  if ((abs(tensor(i,j))/diagonals[i]).real() < tol) tensorChop(i,j) *= 0;
	  if ((abs(tensor(j,i))/diagonals[i]).real() < tol) tensorChop(j,i) *= 0;
      }
    }
	    
    return tensorChop;
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_CHOP_TPP
