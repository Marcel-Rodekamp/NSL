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
        template<typename Type, typename RealType>
        NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type,RealType> & matrix,  const NSL::Tensor<Type,RealType> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        template<typename Type, typename RealType>
        NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type,RealType> & matrix,  const NSL::Tensor<RealType,RealType> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        template<typename Type, typename RealType>
        NSL::Tensor<Type> mat_vec(const NSL::Tensor<RealType,RealType> & matrix,  const NSL::Tensor<Type,RealType> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
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
