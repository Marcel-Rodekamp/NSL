#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
#include <functional>


namespace NSL{
    namespace LinAlg {
        // =====================================================================
        // Product
        // =====================================================================

        //Tensor x Tensor
        template<typename Type>
        NSL::Tensor<Type> mat_vec( const NSL::Tensor<Type> & matrix, const NSL::Tensor<Type> & vector){
            NSL::Tensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //Tensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(const NSL::Tensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){

            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //TimeTensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec( const NSL::TimeTensor<Type> & matrix, const NSL::TimeTensor<Type> & vector){
            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //TimeTensor x Tensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(const NSL::TimeTensor<Type> & matrix, const NSL::Tensor<Type> & vector){
            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //Tensor x scalar.
        template<typename Type>
        NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type> & matrix, const Type & num){
            NSL::Tensor<Type> aux(matrix);
            aux*num;
            return aux;
        }

        //scalar x Tensor.
        template<typename Type>
        NSL::Tensor<Type> mat_vec(const Type & num, const NSL::Tensor<Type> & matrix){
            NSL::Tensor<Type> aux(matrix);
            aux*num;
            return aux;
        }

        //TimeTensor x scalar.
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(const NSL::TimeTensor<Type> & matrix, const Type & num){
            NSL::TimeTensor<Type> aux(matrix);
            aux*num;
            return aux;
        }

        //scalar x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(Type & num, NSL::TimeTensor<Type> & matrix){
            NSL::TimeTensor<Type> aux(matrix);
            aux*num;
            return aux;
        }

        // =====================================================================
        // Exponential
        // =====================================================================

        //exponential(tensor)
        template<typename Type>
        NSL::Tensor<Type> exp (Tensor<Type> & tensor){
            NSL::Tensor<Type> aux(tensor);
            aux.data_= torch::exp(aux.data_);
            return aux;
        }

        //exponential(Timetensor)
        template<typename Type>
        NSL::TimeTensor<Type> exp (TimeTensor<Type> & tensor){
            NSL::TimeTensor<Type> aux(tensor);
            aux.data_= torch::exp(aux.data_);
            return aux;
        }

        // =====================================================================
        // Expansion
        // =====================================================================

        //Expansion of a Tensor
        template<typename Type>
        NSL::Tensor<Type> expand(const Tensor<Type> & tensor, std::deque<long int> & dims){
            NSL::Tensor<Type> aux;
            std::for_each(tensor.data_.sizes().rbegin(), tensor.data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            aux.data_ = (tensor.data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return aux;
        }

        //expantion of a TimeTensor
        template<typename Type>
        NSL:: TimeTensor<Type> expand(const TimeTensor<Type> & tensor, std::deque<long int> & dims){
                NSL::TimeTensor<Type> aux;
                std::for_each(tensor.data_.sizes().rbegin(), tensor.data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
                aux.data_ = (tensor.data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
                return aux;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift, const Type & boundary){
            NSL::Tensor<Type> aux(tensor);
            aux.data_=torch::roll(aux.data_,shift,0);

            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                    aux.data_[i].data() *=boundary;
                }
            }
            else{
                for (int i = (aux.data_).dim() + shift; i < (aux.data_).dim(); i++){
                    aux.data_[i].data() *=boundary;
                }
            }
            return aux;
        }

        //Shift TimeTensor
        template<typename Type>
        NSL::Tensor<Type>  shift( const NSL::Tensor<Type> & tensor, const long int & shift){
            NSL::TimeTensor<Type> aux(tensor);
            aux.data_=torch::roll(aux.data_,shift,0);
            return aux;
        }

        //Shift TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type>  shift(const  NSL::TimeTensor<Type> & tensor, const long int & shift, const Type & boundary){
            NSL::TimeTensor<Type> aux(tensor);
            aux.data_=torch::roll(aux.data_,shift,0);
            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                    aux.data_[i].data() *=boundary;
                }
            }
            else{
                for (int i = (aux.data_).dim() + shift; i < (aux.data_).dim(); i++){
                    aux.data_[i].data() *= boundary;
                }
            }
            return aux;
        }

        //Shift TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type>  shift( NSL::TimeTensor<Type> & tensor, const long int & shift){
            NSL::TimeTensor<Type> aux(tensor);
            aux.data_=torch::roll(aux.data_,shift,0);
            return aux;
        }

        // =====================================================================
        // For each Timeslice
        // =====================================================================

        //Foreach timeslice function:
        template<typename Type>
         auto foreach_timeslice(
//        std::function<NSL::TimeTensor<Type>(NSL::TimeTensor < Type> &, Type &)>functor,
                NSL::TimeTensor<Type> & left,
                NSL::TimeTensor<Type> & right)
                {
                NSL::TimeTensor out(left);
                for(int t = 0; t < left.shape(0); ++t){
                    auto out1=left[t];
                    auto out2 = right[{t}];
                    out[t] = NSL::LinAlg::mat_vec(out1,out2);
                }
                return out;
        }

    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP