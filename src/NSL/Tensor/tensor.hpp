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
//About view (underlying data functions):
namespace NSL {
    template<typename Type>
    class Tensor {
    private:
//        torch::Tensor data_;
    public:
        torch::Tensor data_; //just to prove cout's in the main program.

        //Default constructor.
        explicit Tensor() = default;

        //Construct 1D tensor (i.e. array).
        explicit Tensor(const std::size_t & size) {
            data_ = torch::rand({static_cast<const long int>(size)}, caffe2::TypeMeta::Make<Type>());
        }

        //Construct data_ with sizes stored in dims.
        explicit Tensor(const std::vector<long int> &  dims) {
            data_ = torch::rand(dims, caffe2::TypeMeta::Make<Type>());
        }

        //Copy constructor.
        //ToDo: More efficient data_=other.data_.detach().clone(); or just data_= other.data_;?
        Tensor(Tensor<Type> &other) {
            data_ = other.data_.detach().clone();
        }

        //Random access operator.
        //ToDo: Must be reference to return.
        //ToDo: Accept std::vector
        torch::Tensor operator[](const std::initializer_list<TensorIndex>  & arr){
//            std::initializer_list<TensorIndex> arr;
//            std::copy(dims.begin(), dims.end(), &arr);
            return data_.index(arr).data();
        }

        //Product by a scalar.
        NSL::Tensor<Type> & operator*(const Type & factor){
            data_.data() *= factor;
            return *this;
        }

        //Return size of dimension in data_.
        std::size_t shape(const std::size_t & dim) {
            return data_.size(static_cast<const long int>(dim));
        }

        //Tensor exponential.
        NSL::Tensor<Type> & exp() {
           data_.data()=torch::exp(data_.data());
            return *this;
        }

        //Tensor Expand.
        NSL::Tensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        // return the t-shifted version of this tensor.
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

        //Default constructor.
        explicit TimeTensor() = default;

        //Construct 1D tensor (i.e. array).
        explicit TimeTensor(const std::size_t & size){
            data_ = torch::rand({static_cast<const long int>(size)}, caffe2::TypeMeta::Make<Type>());
        }

        //Construct data_ with sizes stored in dims.
        explicit TimeTensor(const std::vector<long int>  & dims){
           data_ = torch::rand(dims, caffe2::TypeMeta::Make<Type>());
        }

        //Copy constructor.
        explicit TimeTensor(Tensor<Type> & other){
            data_ = other.data_.detach().clone();
        }
        //Random access operator. Return size of dimension in data_.
        torch::Tensor operator[](const std::initializer_list<TensorIndex>  & arr){
            return data_.index(arr).data();
        }

        //Product by a scalar.
        NSL::TimeTensor<Type> & operator*(const Type & factor){
            data_.data() *= factor;
            return *this;
        }

        NSL::TimeTensor<Type> & operator*(const NSL::Tensor<Type> & tensor){
            data_=torch::matmul(data_.data(), tensor.data_.data());
            return *this;
        }

        std::size_t shape(const std::size_t & dim){
            return data_.size(static_cast<const long int>(dim));
        }

        //Tensor exponential.
        //ToDo: implement data_.data()?
       NSL::TimeTensor<Type> & exp() {
            data_.data() = torch::exp(data_.data());
            return *this;
        }
        //Tensor Expand.
        NSL::TimeTensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return *this;
        }

        //Tensor Expand.
        NSL::TimeTensor<Type> & expand(const std::vector<long int> & dimension){
            std::deque<long int> dims;
            std::for_each(dimension.rbegin(), dimension.rend(), [& dims](const long int & x){dims.push_front(x);});
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return *this;
        }

        NSL::TimeTensor<Type> & expand(const std::size_t & dimension){
            std::deque<long int> dims;
            dims.push_front(dimension);
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()}));
            return *this;
        }

        // return the t-shifted version of this tensor.
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
