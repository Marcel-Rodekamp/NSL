#ifndef NSL_LINALG_MAT_TRANS_HPP
#define NSL_LINALG_MAT_TRANS_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> NSL::TimeTensor<Type> mat_transpose(const NSL::TimeTensor<Type> & t){
    NSL::TimeTensor<Type> out(t);
    return out.transpose();
}


} // namespace LinAlg
} // namespace NSL

#endif