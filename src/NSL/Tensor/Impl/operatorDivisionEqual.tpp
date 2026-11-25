#ifndef NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
#define NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP

#include "base.tpp"
#include "../../typePromotion.hpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorDivisionEqual:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    // Compound division: in place, the result keeps the type of this tensor and shares its data.
    // The right operand may be of any number type whose kind (bool < integral < floating point < complex) is not
    // higher than that of Type (NSL::Concept::isKindPreserving); a scalar is converted to Type, a tensor operand is
    // combined as torch does (in the common type, the result rounded to Type). A complex operand of a real tensor
    // or a floating point operand of an integral tensor does not compile.
    // An integral tensor is divided as in C++, truncated towards zero; a bool tensor cannot be divided.
    NSL::Tensor<Type> operator/=(const Type & other) requires (!std::is_same_v<Type,bool>) {
        divide_(other);
        return NSL::Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
        requires (NSL::Concept::isKindPreserving<OtherType,Type> && !std::is_same_v<Type,bool>)
    NSL::Tensor<Type> operator/=(const OtherType & other){
        divide_(static_cast<Type>(other));
        return NSL::Tensor<Type>(this);
    }

    NSL::Tensor<Type> operator/=(const NSL::Tensor<Type> & other) requires (!std::is_same_v<Type,bool>) {
        divide_(torch::Tensor(other));
        return NSL::Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
        requires (NSL::Concept::isKindPreserving<OtherType,Type> && !std::is_same_v<Type,bool>)
    NSL::Tensor<Type> operator/=(const NSL::Tensor<OtherType> & other){
        divide_(torch::Tensor(other));
        return NSL::Tensor<Type>(this);
    }

    // An operand of a higher kind, or any division of a bool tensor, is rejected here, at compile time. Without
    // these exact matches, the same-type overloads above would accept it through an implicit conversion to Type
    // or NSL::Tensor<Type> (e.g. a float tensor /= a complex<float> tensor, or an int tensor /= a double).
    template<NSL::Concept::isNumber OtherType>
        requires (!NSL::Concept::isKindPreserving<OtherType,Type> || std::is_same_v<Type,bool>)
    NSL::Tensor<Type> operator/=(const OtherType & other) = delete;

    template<NSL::Concept::isNumber OtherType>
        requires (!NSL::Concept::isKindPreserving<OtherType,Type> || std::is_same_v<Type,bool>)
    NSL::Tensor<Type> operator/=(const NSL::Tensor<OtherType> & other) = delete;

    private:

    //! data_ /= divisor (a scalar of Type or a torch::Tensor): true division, truncated for integral Type
    template<typename Divisor>
    void divide_(const Divisor & divisor){
        if constexpr (std::is_integral_v<Type>) {
            this->data_.div_(divisor, "trunc");
        } else {
            this->data_.div_(divisor);
        }
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
