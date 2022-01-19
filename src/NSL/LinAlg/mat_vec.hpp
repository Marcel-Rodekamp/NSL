#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
#include <functional>

#include "Tensor/tensor.hpp"


namespace NSL{
    namespace LinAlg {
        // =====================================================================
        // Product mat-vec
        // =====================================================================

        //Tensor x Tensor
        template<typename MatrixType, typename VectorType>
        auto mat_vec(const NSL::Tensor<MatrixType> & matrix,  const NSL::Tensor<VectorType> & vector){

            //! \todo: find a better code for this type deduction
            if constexpr(NSL::is_complex<MatrixType>()){
                if constexpr( ! NSL::is_complex<VectorType>() ) {
                    return NSL::Tensor<MatrixType,VectorType>(
                            torch::matmul(to_torch(matrix), to_torch(static_cast<NSL::Tensor<MatrixType>>(vector))));
                }
            } else if constexpr (NSL::is_complex<VectorType>()) {
                return NSL::Tensor<VectorType,MatrixType>(torch::matmul(to_torch(static_cast<NSL::Tensor<VectorType>>(matrix)),to_torch(vector)));
            } else {

                return NSL::Tensor<MatrixType, typename RT_extractor<MatrixType>::value_type>(
                        torch::matmul(to_torch(matrix), to_torch(vector)));
            }
        }

//        //Tensor x TimeTensor
//        template<typename Type>
//        NSL::TimeTensor<Type> mat_vec(const NSL::Tensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){
//            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
//            return aux;
//        }

//        //TimeTensor x TimeTensor
//        template<typename Type>
//        NSL::TimeTensor<Type> mat_vec( const NSL::TimeTensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){
//            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
//            return aux;
//        }

//        //TimeTensor x Tensor
//        template<typename Type>
//        NSL::TimeTensor<Type> mat_vec(const NSL::TimeTensor<Type> & matrix, const NSL::Tensor<Type> & vector){
//            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
//            return aux;
//        }

        // =====================================================================
        // Expansion
        // =====================================================================

        //Expansion of a Tensor
        template<typename Type>
        NSL::Tensor<Type> expand(const Tensor<Type> & tensor, std::deque<long int> & dims){
            NSL::Tensor<Type> aux;
            aux.copy(tensor);
            aux.expand(dims);
            return aux;
        }

//        //Expansion of a TimeTensor
//        template<typename Type>
//        NSL:: TimeTensor<Type> expand(const TimeTensor<Type> & tensor, std::deque<long int> & dims){
//                NSL::TimeTensor<Type> aux;
//                aux.copy(tensor);
//                aux.expand(dims);
//                return aux;
//        }

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

//        //Shift TimeTensor
//        template<typename Type>
//        NSL::TimeTensor<Type>  shift(const  NSL::TimeTensor<Type> & tensor, const long int & shift, const Type & boundary){
//            NSL::TimeTensor<Type> aux;
//            aux.copy(tensor);
//            aux.shift(shift, boundary);
//            return aux;
//        }

        //Shift TimeTensor
//        template<typename Type>
//        NSL::TimeTensor<Type>  shift( NSL::TimeTensor<Type> & tensor, const long int & shift){
//            NSL::TimeTensor<Type> aux;
//            aux.copy(tensor);
//            aux.shift(shift);
//            return aux;
//        }
        //Identity matrix
//        template<typename Type>
//        torch::Tensor identity(const Type & length){
            // Note: This should be a standard normal N(mean=0,var=1) distribution
            
        
//          return (torch::eye(length,torch::TensorOptions().dtype<Type>()));
       // }
    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP