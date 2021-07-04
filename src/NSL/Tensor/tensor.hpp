#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

#include <memory>

// ============================================================================
// Declarations

namespace NSL{

template<typename Type, bool GPU> class Tensor;

} // namespace NSL

// ============================================================================
// CPU Implementations
namespace NSL {

template<typename Type>
class Tensor<Type, false> {
    private:
        std::shared_ptr <Type> _ptr;

    public:
        explicit Tensor(std::size_t size):
            _ptr( new Type[size] )
        {}

        Type &operator[](std::size_t idx) {
            return _ptr.get()[idx];
        }
};

} // namespace

#endif //NSL_TENSOR_HPP
