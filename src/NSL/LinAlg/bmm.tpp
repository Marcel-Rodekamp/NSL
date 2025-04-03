#ifndef NSL_LINALG_BMM_HPP
#define NSL_LINALG_BMM_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{


<<<<<<< HEAD
//! matrix @ matrix / tensor @ tensor
/*!
 * Implementation is the same as `NSL::LinAlg::mat_vec` 
=======
//! tensor @ tensor
/*!
 * Batch matrix multiplication
>>>>>>> 3eb6c0b (Added bmm sausage computation, bmmeval, bmmgrad, bmmlogdetM for hubbardexp and test executable)
 * */

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> bmm(const NSL::Tensor<Type> & leftTensor, const NSL::Tensor<Type> & rightTensor){
    return torch::bmm(leftTensor,rightTensor);
}
        
} // namespace NSL::LinAlg

#endif //NANOSYSTEMLIBRARY_BMM_HPP
