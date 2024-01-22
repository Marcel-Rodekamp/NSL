#ifndef NSL_RNG_CHOICE_TPP
#define NSL_RNG_CHOICE_TPP

#include "../concepts.hpp"

namespace NSL::Random{

//! Generates a random integer from 0 to a with weights given by weights
template<NSL::Concept::isNumber Type>
NSL::size_t choice( NSL::size_t a, NSL::Tensor<Type> weights){
    // multinomial draws a random integer between 0 and len(weights). With 
    // a probability given in weights
    NSL::Tensor<NSL::size_t> res = torch::multinomial(weights,1);

    // we just need to return the single integer generated above;
    return res(0);
}

} // namespace NSL::Random

#endif // NSL_RNG_CHOICE_TPP
