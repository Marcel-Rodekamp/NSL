#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
//ToDo:
//ToDo: Make private the data? Or it is slower. Then must change the copy to a copy in other tensor.
namespace NSL{
    namespace LinAlg {
        //ToDo: What is more efficient, do a copy and then follow as tensor.hpp or create a data_ empty.
        //Tensor x Tensor
        template<typename Type>
        NSL::Tensor<Type> mat_vec(NSL::Tensor<Type> & matrix, NSL::Tensor<Type> & vector){
            NSL::Tensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }
        //Tensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(NSL::Tensor<Type> & matrix, NSL::TimeTensor<Type> & vector){
            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //TimeTensor x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(NSL::TimeTensor<Type> & matrix, NSL::TimeTensor<Type> & vector){
            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }

        //TimeTensor x Tensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(NSL::TimeTensor<Type> & matrix, NSL::Tensor<Type> & vector){
            NSL::TimeTensor<Type> aux;
            aux.data_ =torch::matmul(matrix.data_, vector.data_);
            return aux;
        }
        //Tensor x scalar.
        template<typename Type>
        NSL::Tensor<Type> mat_vec(NSL::Tensor<Type> & matrix, const Type & num){
            NSL::Tensor<Type> aux;
            aux.data_ *=num;
            return aux;
        }
        //scalar x Tensor.
        template<typename Type>
        NSL::Tensor<Type> mat_vec(const Type & num, NSL::Tensor<Type> & matrix){
            NSL::Tensor<Type> aux;
            aux.data_ =num*aux.data_;
            return aux;
        }
        //TimeTensor x scalar.
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(NSL::TimeTensor<Type> & matrix, const Type & num){
            NSL::TimeTensor<Type> aux;
            aux.data_ =matrix.data_*num;
            return aux;
        }
        //scalar x TimeTensor
        template<typename Type>
        NSL::TimeTensor<Type> mat_vec(Type & num, NSL::TimeTensor<Type> & matrix){
            NSL::TimeTensor<Type> aux;
            aux.data_ =num*matrix.data_;
            return aux;
        }

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

        //Expansion of a Tensor
        template<typename Type>
        NSL::Tensor<Type> expand(Tensor<Type> & tensor, std::deque<long int> & dims){
            NSL::Tensor<Type> aux;
            std::for_each(tensor.data_.sizes().rbegin(), tensor.data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            aux.data_ = (tensor.data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return aux;
        }

        //expantion of a TimeTensor
        template<typename Type>
        NSL:: TimeTensor<Type> expand(TimeTensor<Type> & tensor, std::deque<long int> & dims){
                NSL::TimeTensor<Type> aux;
                std::for_each(tensor.data_.sizes().rbegin(), tensor.data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
                aux.data_ = (tensor.data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
                return aux;
        }

        //Shift Tensor
        template<typename Type>
        NSL::Tensor<Type>  shift( NSL::Tensor<Type> & tensor, const long int & shift, const Type & boundary){
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
        NSL::TimeTensor<Type>  shift( NSL::TimeTensor<Type> & tensor, const long int & shift, const Type & boundary){
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

        template<typename Type>
        auto foreach_timeslice(
                std::function <NSL::TimeTensor<Type> (NSL::TimeTensor<Type> & , NSL::TimeTensor<Type>&)> functor,
                NSL::TimeTensor<Type> & left,
                NSL::TimeTensor<Type> & right)
                {
                NSL::TimeTensor out(left);
                for(int t = 0; t < left.shape(0); ++t){
                    out[{t}] = functor(left[{t}],right[{t}]);
                    }
                }
    } // namespace LinAlg
} // namespace NSL
/*expand ()*/
#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP