
//
// Created by marcel on 8/19/21.
//

#ifndef NANOSYSTEMLIBRARY_COMPLEX_HPP
#define NANOSYSTEMLIBRARY_COMPLEX_HPP

#include "torch/torch.h"

namespace NSL{
    // hide the use of special complex implementations as it is specific to the underlying library
    template<typename Type>
    using complex = c10::complex<Type>;

}

#endif //NANOSYSTEMLIBRARY_COMPLEX_HPP