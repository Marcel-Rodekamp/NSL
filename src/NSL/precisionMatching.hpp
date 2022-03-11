#ifndef NSL_PRECISION_MATCHING_HPP
#define NSL_PRECISION_MATCHING_HPP

#include <limits>
#include "complex.hpp"


//! \file precisionMatching.hpp

namespace NSL {

//! Determine the more precise type
/*! 
 * In mixed precision determinantion it is often required to identify the more
 * precise type of a template if it can not be known from the beginning.
 * This struct can be used to identify this. Given two types, e.g. float 
 * and double, the more precise type of these can be identified by
 * ```
 *  using precise_type = Precise<T1,T2>::type;
 * ```
 * 
 * - Comparison between integer and floating point will always return the floating point.
 *   ```
 *   Precise<float,int>::type => float
 *   ```
 * - If one of the type is complex valued, the determined type will be complex
 *   with more precise floating point
 *   ```
 *   Precise<float,NSL::complex<float>>::type => NSL::complex<float>
 *   Precise<couble,NSL::complex<float>>::type => NSL::complex<double>
 *   ```
 * - Considering issue #33, no rule is applied to different integral types at
 *   the current implementation.
 *
 * */
template<typename T1,typename T2> 
struct Precise {
    static constexpr auto compare_precision(){

        // compare integer with floating point
        if constexpr (std::is_integral_v<T1> && !std::is_integral_v<T2>){
            // the floating point is more precise
            return std::numeric_limits<T2>::min();
        // compare floating point with integer
        } else if constexpr (!std::is_integral_v<T1> && std::is_integral_v<T2>){
            // the floating point is more precise
            return std::numeric_limits<T1>::min();
        // compare floating points
        } else {
            // the floating point number with smallest machine precision is more precise
            if constexpr (std::numeric_limits<T1>::epsilon() < std::numeric_limits<T2>::epsilon() ){
                return std::numeric_limits<T1>::min();
            } else {
                return std::numeric_limits<T2>::min(); 
            }
        }
    }
    using type = decltype(compare_precision());
};

template<typename T1,typename T2> struct Precise<NSL::complex<T1>,T2>{using type = NSL::complex<typename Precise<T1,T2>::type>;};
template<typename T1,typename T2> struct Precise<T1,NSL::complex<T2>>{using type = NSL::complex<typename Precise<T1,T2>::type>;};
template<typename T1,typename T2> struct Precise<NSL::complex<T1>,NSL::complex<T2>>{using type = NSL::complex<typename Precise<T1,T2>::type>;};

//! Determine the less precise type
/*! 
 * In mixed precision determinantion it is often required to identify the less
 * precise type of a template if it can not be known from the beginning.
 * This struct can be used to identify this. Given two types, e.g. float 
 * and double, the less precise type of these can be identified by
 * ```
 *  using imprecise_type = Imprecise<T1,T2>::type;
 * ```
 * 
 * - Comparison between integer and floating point will always return the integer.
 *   ```
 *   Precise<float,int>::type => int
 *   ```
 * - If one of the type is complex valued, the determined type will be complex
 *   with less precise floating point
 *   ```
 *   Precise<float,NSL::complex<float>>::type => NSL::complex<float>
 *   Precise<couble,NSL::complex<float>>::type => NSL::complex<float>
 *   ```
 * - Considering issue #33, no rule is applied to different integral types at
 *   the current implementation.
 *
 * */
template<typename T1,typename T2> 
struct Imprecise {
    static constexpr auto compare_precision(){
        // compare integer with floating point
        if constexpr (std::is_integral_v<T1> && !std::is_integral_v<T2>){
            // the floating point is more precise
            return std::numeric_limits<T1>::min();
        // compare floating point with integer
        } else if constexpr (!std::is_integral_v<T1> && std::is_integral_v<T2>){
            // the floating point is more precise
            return std::numeric_limits<T2>::min();
        // compare floating points
        } else {
            // the floating point number with smallest machine precision is more precise
            if constexpr (std::numeric_limits<T1>::epsilon() > std::numeric_limits<T2>::epsilon() ){
                return std::numeric_limits<T1>::min();
            } else {
                return std::numeric_limits<T2>::min(); 
            }
        }
    }
    using type = decltype(compare_precision());
};

template<typename T1,typename T2> struct Imprecise<NSL::complex<T1>,T2>{using type = NSL::complex<typename Imprecise<T1,T2>::type>;};
template<typename T1,typename T2> struct Imprecise<T1,NSL::complex<T2>>{using type = NSL::complex<typename Imprecise<T1,T2>::type>;};
template<typename T1,typename T2> struct Imprecise<NSL::complex<T1>,NSL::complex<T2>>{using type = NSL::complex<typename Imprecise<T1,T2>::type>;};

} // namespace NSL

#endif // NSL_PRECISION_MATCHING_HPP
