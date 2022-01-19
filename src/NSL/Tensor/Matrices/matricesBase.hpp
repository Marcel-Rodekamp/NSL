#ifndef NSL_TENSOR_MATRICES_BASE_HPP
#define NSL_TENSOR_MATRICES_BASE_HPP

#include "../tensor.hpp"
#include "../../assert.hpp"
#include "../../complex.hpp"

namespace NSL::Matrices {

template <typename Type>
class Matrices: public Tensor<Type> {
    using size_t = int64_t;

    public:
    


    //template<typename... SizeType>
    Matrices(/*const size_t & size0, const SizeType &... sizes*/) : Tensor<Type>(/*size0, sizes...*/)
        
    {}

    //functions

    NSL::Tensor<Type> Identity(const size_t & size ) {
        this->data_ = torch::eye(size, torch::TensorOptions().dtype<Type>());
        return (*this);
    }



};
}
#endif
