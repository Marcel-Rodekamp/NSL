#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP
#include<torch/torch.h>
#include <memory>
#include<vector>
#include <iomanip>
#include <complex>
#include <cmath>
using namespace torch::indexing;
// ============================================================================
// CPU Implementations
//About underlying data: https://stackoverflow.com/questions/62523708/pointer-type-behavior-in-pytorch
namespace NSL {
    template<typename Type>
    class Tensor {
    private:
//        torch::Tensor data_;
    public:
        torch::Tensor data_; //just to prove cout's in the main program.

        //Default constructor
        explicit Tensor() = default;

        // construct 1D tensor (i.e. array)
        explicit Tensor(std::size_t & size) {
            data_ = torch::zeros({static_cast<const long int>(size)}, caffe2::TypeMeta::Make<Type>());
        }

        // construct data_ with sizes stored in dims
        explicit Tensor(const std::vector<long int> &  dims) {
            data_ = torch::zeros(dims, caffe2::TypeMeta::Make<Type>());
        }

        // copy constructor
        //ToDo: More efficient detach clone or just data_= ...?
        Tensor(Tensor<Type> &other) {
            data_ = other.data_.detach().clone();
        }

        // random access operator
        //ToDo: Must be reference to return.
        //ToDo: Accept std::vector
        torch::Tensor operator[](const std::initializer_list<TensorIndex>  & arr){
//            std::initializer_list<TensorIndex> arr;
//            std::copy(dims.begin(), dims.end(), &arr);
            return data_.index(arr).data();
        }

        NSL::Tensor<Type> & operator*(const Type & factor){
            data_.data() *= factor;
            return *this;
        }
        // return size of dimension in data_
        std::size_t shape(const std::size_t & dim) {
            return data_.size(static_cast<const long int>(dim));
        }

        //Tensor exponential
        NSL::Tensor<Type> & exp() {
           data_.data()=torch::exp(data_.data());
            return *this;
        }

        //Tensor Expand
        NSL::Tensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        // return the t-shifted version of this tensor
        //ToDo: What if shift > dimension?
        //Why use roll? https://discuss.pytorch.org/t/is-there-a-way-to-create-a-rolled-shifted-and-wrapped-view-of-a-tensor/67568
        NSL::Tensor<Type> & shift(const long int & shift, const Type & boundary){

            data_.data() = torch::roll(data_.data(),shift,0);
            std::cout<<"Boundary:" <<boundary<<std::endl;
            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                    (data_[i]).data() *=boundary;
                }
            }
            else{
                for (int i = (data_).dim() + shift; i < (data_).dim(); i++){
                    (data_[i]).data() *=boundary;
                }
            }
            return *this;
        }

        //Print tensor.
        void print (){
            std::cout<<data_ <<std::endl;
        }

        //Print tensor complex.
        void print_complex (){
            std::cout<< torch::view_as_real(data_)<<std::endl;
        }
    }; //Tensor class
//========================================================================
//Time Tensor Class

    template<typename Type>
    class TimeTensor {
    private:
//                torch::Tensor data_;
    public:
        torch::Tensor data_;//just to prove cout's in the main program.

        //Default constructor
        explicit TimeTensor() = default;

        // construct 1D tensor (i.e. array)
        explicit TimeTensor(std::size_t & size){
            data_ = torch::zeros({static_cast<const long int>(size)}, caffe2::TypeMeta::Make<Type>());
        }

        // construct data_ with sizes stored in dims
        explicit TimeTensor(const std::vector<long int>  & dims){
           data_ = torch::zeros(dims, caffe2::TypeMeta::Make<Type>());
        }

        // copy constructor
        explicit TimeTensor(Tensor<Type> & other){
            data_ = other.data_.detach().clone();
        }
        // random access operator. Return size of dimension in data_
        torch::Tensor operator[](const std::initializer_list<TensorIndex>  & arr){
            //            std::initializer_list<TensorIndex> arr;
            //            std::copy(dims.begin(), dims.end(), &arr);
            return data_.index(arr).data();
        }
        NSL::TimeTensor<Type> & operator*(const Type & factor){
            data_.data() *= factor;
            return *this;
        }

        std::size_t shape(const std::size_t & dim){
            return data_.size(static_cast<const long int>(dim));
        }

        //Tensor exponential
        //ToDo: implement data_.data()?
       NSL::TimeTensor<Type> & exp() {
            data_.data() = torch::exp(data_.data());
            return *this;
        }
        //Tensor Expand
        NSL::TimeTensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return *this;
        }

        // return the t-shifted version of this tensor
        //ToDo: What if shift > dimension?
        //Why use roll? https://discuss.pytorch.org/t/is-there-a-way-to-create-a-rolled-shifted-and-wrapped-view-of-a-tensor/67568
        NSL::TimeTensor<Type> & shift(const long int & shift, const Type & boundary){

            data_.data()= torch::roll(data_.data(),shift,0);;
            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                      data_[i].data() *= boundary;
                }
            }
            else{
                for (int i = (data_).dim() + shift; i < (data_).dim(); i++){
                    data_[i].data() *= boundary;
                }
            }
            return (*this);
        }

        //Print tensor.
        void print (){
            std::cout<<data_ <<std::endl;
        }

        //Print tensor complex.
        void print_complex (){
            std::cout<< torch::view_as_real(data_)<<std::endl;
        }

    }; //Time Tensor class
} // namespace NSL

#endif //NSL_TENSOR_HPP
