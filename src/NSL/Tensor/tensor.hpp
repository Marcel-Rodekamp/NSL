#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

#include <memory>

// ============================================================================
// CPU Implementations
namespace NSL {

template<typename Type>
class Tensor {
    private:
        // xt::xtensor data_;

    public:
        // construct 1D tensor (i.e. array)
        explicit Tensor(std::size_t size);

        // construct data_ with sizes stored in dims
        explicit Tensor(std::initializer_list<std::size_t> dims);

        // copy constructor
        Tensor(Tensor<Type> & other);

        // random access operator
        Type &operator()(std::size_t idx);

        // return size of dimension in data_
        const std::size_t shape(const std::size_t dim);
};

template<typename Type>
class TimeTensor {
    private:
    // xt::xtensor data_;

    public:
    // construct 1D tensor (i.e. array)
    explicit TimeTensor(std::size_t size);

    // construct data_ with sizes stored in dims
    explicit TimeTensor(std::initializer_list<std::size_t> dims);

    // copy constructor
    TimeTensor(Tensor<Type> & other);

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
