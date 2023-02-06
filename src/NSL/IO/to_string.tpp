#ifndef NSL_IO_TO_STRING_TPP
#define NSL_IO_TO_STRING_TPP
#include "../complex.hpp"
#include "../concepts.hpp"

namespace NSL {
template <NSL::Concept::isNumber Type>
inline std::string to_string(const Type &z){
    if constexpr(NSL::is_complex<Type>()){
        auto re = NSL::real(z);
        auto im = NSL::imag(z);
        if (im < 0){
            return std::to_string(re)+std::to_string(im)+"i";
        }
        return std::to_string(re)+"+"+std::to_string(im)+"i";
    } else {
        return std::to_string(z); 
    }
}
} // namespace NSL

#endif // NSL_IO_TO_STRING_TPP
