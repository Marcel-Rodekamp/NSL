#ifndef NANOSYSTEMLIBRARY_MAT_MUL_HPP
#define NANOSYSTEMLIBRARY_MAT_MUL_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
#include <functional>


namespace NSL{
    namespace LinAlg {

        //! matrix multiplication
        /*!
         * out = left @ right 
         * */
        template<typename T>
        NSL::Tensor<T> mat_mul(const NSL::Tensor<T> & left, const NSL::Tensor<T> & right){
            return NSL::Tensor<T>( torch::matmul( to_torch(left), to_torch(right) ) );    
        }
        
    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_MAT_MUL_HPP
