#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
#include <functional>


namespace NSL{
    namespace LinAlg {
        // =====================================================================
        // Product mat-vec
        // =====================================================================

        //Tensor x Tensor
        template<typename Type>
        NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type> & matrix,  const NSL::Tensor<Type> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        //Tensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(const NSL::Tensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        //TimeTensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec( const NSL::TimeTensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        //TimeTensor x Tensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(const NSL::TimeTensor<Type> & matrix, const NSL::Tensor<Type> & vector){
            NSL::Tensor<Type> aux(torch::matmul(to_torch(matrix), to_torch(vector)));
            return aux;
        }

        // =====================================================================
        // Exponential
        // =====================================================================

        //exponential(tensor)
        template<typename Type>
        NSL::Tensor<Type> exp (Tensor<Type> & tensor){
            NSL::Tensor<Type> aux;
            aux.copy(tensor);
            aux.exp();
            return aux;
        }

        //exponential(Timetensor)
        template<typename Type>
        NSL::TimeTensor<Type> exp (TimeTensor<Type> & tensor){
            NSL::TimeTensor<Type> aux;
            aux.copy(tensor);
            aux.exp();
            return aux;
        }

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

        //Expansion of a TimeTensor
        template<typename Type>
        NSL:: TimeTensor<Type> expand(const TimeTensor<Type> & tensor, std::deque<long int> & dims){
                NSL::TimeTensor<Type> aux;
                aux.copy(tensor);
                aux.expand(dims);
                return aux;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift, const Type & boundary){
            NSL::Tensor<Type> aux;
            aux.copy(tensor);
            aux.shift(shift, boundary);
            return aux;
        }

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift){
            NSL::Tensor<Type> aux;
            aux.copy(tensor);
            aux.shift(shift);
            return aux;
        }

        //Shift TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type>  shift(const  NSL::TimeTensor<Type> & tensor, const long int & shift, const Type & boundary){
            NSL::TimeTensor<Type> aux;
            aux.copy(tensor);
            aux.shift(shift, boundary);
            return aux;
        }

        //Shift TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type>  shift( NSL::TimeTensor<Type> & tensor, const long int & shift){
            NSL::TimeTensor<Type> aux;
            aux.copy(tensor);
            aux.shift(shift);
            return aux;
        }

    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP