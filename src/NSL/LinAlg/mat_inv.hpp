#ifndef NSL_LINALG_MAT_INV_HPP
#define NSL_LINALG_MAT_INV_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> NSL::TimeTensor<Type> mat_inv(const NSL::TimeTensor<Type> & t){
    NSL::TimeTensor<Type> out(torch::inverse(to_torch(t)));
    return out;
}


} // namespace LinAlg
} // namespace NSL

#endif
