#ifndef NSL_TENSOR_CUH
#define NSL_TENSOR_CUH

#ifdef __CUDACC__
#include "tensor.hpp"

namespace NSL {
template<typename Type>
class Tensor<Type, true> {
    private:
        Type * _ptr;

    public:
        explicit Tensor(std::size_t size) {

            cudaMallocManaged(&_ptr, size * sizeof(Type));

        }

        Type &operator[](std::size_t idx) {
            return _ptr[idx];
        }

        Type * data(){
            return _ptr;
        }
};

} // namespace NSL
#endif // __CUDACC__
#endif // NSL_TENSOR_CUH