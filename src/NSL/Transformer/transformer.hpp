#ifndef NSL_Transformer_HPP
#define NSL_Transformer_HPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include <utility>

namespace NSL {

template<NSL::Concept::isNumber Type>
class Transformer {

public:

    //! A function that takes an input configuration applies the transformation and returns 
    //! the result and the log determinant of the associated Jacobian matrix.
    virtual std::pair<NSL::Tensor<Type>, Type> forward(const NSL::Tensor<Type> & input ) = 0;
    virtual std::pair<NSL::Configuration<Type>, Type> forward(NSL::Configuration<Type> & input ) = 0;

    virtual std::pair<NSL::Tensor<Type>, Type> inverse(const NSL::Tensor<Type> & input ) = 0;
    virtual std::pair<NSL::Configuration<Type>, Type> inverse(NSL::Configuration<Type> & input ) = 0;

}; // Transformer 


} // namespace NSL


#endif // NSL_Transformer_HPP
