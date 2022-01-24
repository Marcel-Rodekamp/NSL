#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP

#include<torch/torch.h>
#include<deque>

#include <Tensor/tensor.hpp>


namespace NSL{
    namespace LinAlg {
        // =====================================================================
        // Product mat-vec
        // =====================================================================

        //Tensor x Tensor
        // Matrix Complex, Vector Real => promote Vector
        template<typename MatCType, typename MatRType, typename VecRType>
        NSL::Tensor<MatCType,MatRType>  mat_vec(const NSL::Tensor<MatCType,MatRType> & matrix, const NSL::Tensor<VecRType, VecRType> & vector){
            return NSL::Tensor<MatCType,MatRType> (torch::matmul(
                    matrix, static_cast<const NSL::Tensor<MatCType,MatRType>>(vector)
            ));
        }

        //Tensor x Tensor
        // Matrix Real, Vector Complex => promote Matrix
        template<typename MatRType, typename VecCType, typename VecRType>
        NSL::Tensor<VecCType,VecRType>  mat_vec(const NSL::Tensor<MatRType,MatRType> & matrix, const NSL::Tensor<VecCType, VecRType> & vector){
            return NSL::Tensor<VecCType,VecRType> (torch::matmul(
                    static_cast<const NSL::Tensor<VecCType,VecRType>>(matrix), vector
            ));
        }

        //Tensor x Tensor
        // Matrix Complex/Real, Vector Complex/Real
        template<typename CType, typename RType>
        NSL::Tensor<CType,RType>  mat_vec(const NSL::Tensor<CType,RType> & matrix, const NSL::Tensor<CType, RType> & vector){
            return NSL::Tensor<CType,RType> (torch::matmul(
                    matrix, vector
            ));
        }


    // =====================================================================
        // Expansion
        // =====================================================================

        //Expansion of a Tensor
        template<typename Type>
        NSL::Tensor<Type> expand(const Tensor<Type> & tensor, std::deque<long int> & dims){
            NSL::Tensor<Type> aux(tensor);
            aux.expand(dims);
            return aux;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift, const Type & boundary){
            NSL::Tensor<Type> aux(tensor);
            aux.shift(shift, boundary);
            return aux;
        }

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift){
            NSL::Tensor<Type> aux(tensor);
            aux.shift(shift);
            return aux;
        }

    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP
