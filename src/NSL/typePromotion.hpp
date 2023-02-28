#ifndef NSL_TYPE_PROMOTION_HPP
#define NSL_TYPE_PROMOTION_HPP

//! \file

#include "concepts.hpp"
#include "complex.hpp"

namespace NSL{

//forward declar
template<NSL::Concept::isNumber Type> class Tensor;

//! Helper struct to deduce the common type, see `NSL::CommonTypeOf`
template<typename TypeLeft, typename TypeRight>
struct TypePromotion {

    // Deduce the common type from the two given types accoring to the rule
    // The common type is the most general type accessible from the two types
    // see CommonTypeOf below for more details
    // The requires ensures that the Types are C++ bare types (bool,int,float,double,complex<float>,complex<fouble>)
    // for which implicit conversion exists (or complex where we can use the real types for it)
    static auto deduce()
        requires(NSL::Concept::isNumber<TypeLeft> && NSL::Concept::isNumber<TypeRight>)
    {
        if constexpr (std::is_same_v<TypeLeft,TypeRight>) {
            return TypeLeft{}; 
        } else if constexpr( NSL::is_complex<TypeLeft>() || NSL::is_complex<TypeRight>() ){
            // extract the real types and determine the common real type (float/double)
            // using C++ type promotion
            // If either of them is not complex RealTypeOf will just return the 
            // original type
            // If at least one is complex the result should always be complex
            return NSL::complex<
                decltype(
                    NSL::RealTypeOf<TypeLeft>{}+NSL::RealTypeOf<TypeRight>{}
                )>{};
        }  else {
            // utilize the real type promotion from C++
            return TypeLeft{}+TypeRight{};
        }
    };

    using CommonType = decltype(deduce());
};


//! Helper struct to deduce the common type, see `NSL::CommonTypeOf`
/*! specialize for NSL::Tensor where the type deduction is on the scalar type 
 * The common tensor type is simply the tensor of the common scalar type
 */ 
template<typename TypeLeft, typename TypeRight>
struct TypePromotion<NSL::Tensor<TypeLeft>,NSL::Tensor<TypeRight>>{
    using CommonType = NSL::Tensor<decltype(TypePromotion<TypeLeft,TypeRight>::deduce())>;
};

//! Helper struct to deduce the common type, see `NSL::CommonTypeOf`
/*! specialize for NSL::Tensor-scalar
 * * The common tensor type is simply the tensor of the common scalar type
 */ 
template<typename TypeLeft, typename TypeRight>
struct TypePromotion<NSL::Tensor<TypeLeft>,TypeRight>{
    using CommonType = NSL::Tensor<decltype(TypePromotion<TypeLeft,TypeRight>::deduce())>;
};

//! Helper struct to deduce the common type, see `NSL::CommonTypeOf`
/*! specialize for scalar-NSL::Tensor
 * * The common tensor type is simply the tensor of the common scalar type
 */ 
template<typename TypeLeft, typename TypeRight>
struct TypePromotion<TypeLeft,NSL::Tensor<TypeRight>>{
    using CommonType = NSL::Tensor<decltype(TypePromotion<TypeLeft,TypeRight>::deduce())>;
};

//! Deducing a common data type from two (`TypeLeft`, `TypeRight`) types.
/*!
 * When working with different precisions/number types a well defined 
 * rule should be inplace to cast the type.
 * We apply the following rule:
 * 
 * <CENTER><B>
 * The common type of two basic (scalar-)types is the most general type 
 *    that can be formed by them.
 * </B></CENTER>
 *
 *  This means, for the allowed types:
 *  |                 |       bool      |       int       |      float      |      double     | complex<float>  | complex<double> |
 *  | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
 *  |<B>bool           </B>|`      bool     `|`      int      `|`     float     `|`     double    `|` complex<float>`|`complex<double>`|
 *  |<B>int            </B>|`     int       `|`      int      `|`     float     `|`     double    `|` complex<float>`|`complex<double>`| 
 *  |<B>float          </B>|`     float     `|`     float     `|`     float     `|`     double    `|` complex<float>`|`complex<double>`|
 *  |<B>double         </B>|`     double    `|`     double    `|`     double    `|`     double    `|` complex<float>`|`complex<double>`|
 *  |<B>complex<float> </B>|` complex<float>`|` complex<float>`|` complex<float>`|` complex<float>`|` complex<float>`|`complex<double>`|
 *  |<B>complex<double></B>|`complex<double>`|`complex<double>`|`complex<double>`|`complex<double>`|`complex<double>`|`complex<double>`|
 * 
 * This table can be read in the following way. Choose the two operating 
 * types (row & col), then their associated element in the table represents
 * the common type they are cast to.
 *
 * The common type is compliant to the C++ standard implicit conversion of 
 * the basic types --- bool,int,float,double ---
 * and also extends to the complex types.
 *
 * It is also applied for `NSL::Tensor` where the common type of two tensors 
 * is the tensor of common scalar type.
 *
 * The struct `NSL::TypePromotion<TypeLeft,TypeRight>` encodes this rules.
 * */
template<typename TypeLeft, typename TypeRight>
using CommonTypeOf = typename TypePromotion<TypeLeft,TypeRight>::CommonType;

template< typename Type1, typename Type2, typename ... Types_>
static constexpr auto deduceIteration(){
    // stop condition
    if constexpr (sizeof...(Types_) == 0) {
        return CommonTypeOf<Type1,Type2>{};
    } else {
        return CommonTypeOf<Type1, decltype(deduceIteration<Type2,Types_...>)>{};
    }
}

template<typename Type>
static constexpr auto deduceIteration(){
    return Type{};
}

template<typename ... Types>
using CommonTypeOfPack = decltype(deduceIteration<Types...>());


} // namespace NSL
#endif //NSL_TYPE_PROMOTION_HPP
