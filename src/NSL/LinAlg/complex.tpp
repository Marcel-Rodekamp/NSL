#ifndef NSL_LIN_ALG_COMPLEX_HPP
#define NSL_LIN_ALG_COMPLEX_HPP

#include "../complex.hpp"
#include "../concepts.hpp"


namespace NSL {

} //namespace NSL

namespace NSL::LinAlg {

//! Returns the complex conjugate, maintaining type (`complex<>` if `complex<>`, not if not).
template<NSL::Concept::isNumber Type>
inline NSL::RealTypeOf<Type> arg(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::arg(value);
    }
    else {
        if(value > 0) return static_cast<NSL::RealTypeOf<Type>>(0);
        //! todo We should be very careful about the branch-cut of arg.
        // If we want arg to be single-valued, we should pick a finite interval,
        // say (-π,+π], which is 2π periodic.  However, the negative real axis
        // is right on the boundary.
        return static_cast<NSL::RealTypeOf<Type>>(+std::numbers::pi);
        // I picked + because in Mathematica
        //      Arg[-1] == +π
        // and I trust Wolfram to have these conventions sorted out.
        // Other choices are possible, and perhaps even are sensible!
        //! todo check what c10's std::arg does and ensure consistency.
    }
}

} // namespace NSL::LinAlg

#endif //NSL_LIN_ALG_COMPLEX_HPP
