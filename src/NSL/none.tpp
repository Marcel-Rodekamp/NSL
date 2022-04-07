#ifndef NSL_NONE_TYPE_TPP
#define NSL_NONE_TYPE_TPP

#include "torch/torch.h"
#include "types.hpp"

namespace NSL {

struct none_t {
    none_t() = default;

    //! Static conversion to torch::indexing::None
    static const c10::nullopt_t torch() {
        return torch::indexing::None;
    }

    //! implicit conversion to torch::indexing::None
    operator c10::optional<NSL::size_t> () {
        return torch::indexing::None;
    }

    //! implicit conversion to torch::indexing::None
    operator c10::nullopt_t () {
        return torch::indexing::None;
    }
};

// Defining a global variable in a header file causes the linker to see 
// multiplie definitions of that variable.
// Since C++17 we can circumvent this behaviour by using inline.
// For more information consider:
// https://stackoverflow.com/a/38043566
// https://stackoverflow.com/a/52045466
// These types are used for the slicing of a NSL::Tensor. Their exact
// behaviour is defined in src/NSL/sliceObj.tpp
inline none_t None;
inline none_t End;
inline none_t All;

} // namespace NSL

#endif
