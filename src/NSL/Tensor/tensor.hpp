#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>

using namespace torch::indexing;

// ============================================================================
// CPU Implementations
namespace NSL {

template<typename Type>
class Tensor {
    private:
    //torch::Tensor data_;

    public:
        torch::Tensor data_; //just to prove cout's in the main program.

        //Default constructor
        explicit Tensor()= default;

        // construct 1D tensor (i.e. array)
        explicit Tensor(std::size_t size){
            data_ = torch::zeros({static_cast<const long int>(size)});
        }

        // construct data_ with sizes stored in dims
        explicit Tensor(std::vector< long int> dims){
            data_= torch::zeros(dims);
        }

        // copy constructor
        Tensor(Tensor<Type> & other){
            data_ = other.data_;
        }

        // random access operator
        //ToDo: Must be reference to return.
        Type  operator()( std::vector<long int> idx){
            return data_.index(idx).item<Type>();
        }

        // return size of dimension in data_
        std::size_t shape(std::size_t dim){
            return data_.size(static_cast<const long int>(dim));
        }

        //Tensor exp
        Tensor<Type> exp(){
            Tensor<Type> exp;
            exp.data_ = torch::exp(data_);
            return exp;
        }

};

template<typename Type>
class TimeTensor {
    private:
    //        torch::Tensor data_;

    public:
    torch::Tensor data_; //just to prove cout's in the main program.

    // construct 1D tensor (i.e. array)
    explicit TimeTensor(std::size_t size){
        data_ = torch::zeros({static_cast<const long int>(size)});
    };

    // construct data_ with sizes stored in dims
    explicit TimeTensor(std::initializer_list<std::size_t> dims){
    };

    // copy constructor
    TimeTensor(Tensor<Type> & other){
        data_ = other.data_;
    };

    // random access operator
    Type &operator()(std::size_t idx);

    // return size of dimension in data_
    const std::size_t shape(const std::size_t dim);

    // return the t-shifted version of this tensor
    // could we do it with views?
    TimeTensor<Type> & shift(const std::size_t offset);
};

} // namespace

#endif //NSL_TENSOR_HPP
