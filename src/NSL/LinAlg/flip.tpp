#ifndef NSL_LINALG_FLIP_TPP
#define NSL_LINALG_FLIP_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg  {

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> flip( const NSL::Tensor<Type> & t, std::vector<NSL::size_t> dims){
    return torch::flip(
        t,dims
    );

}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_FLIP_TPP