#ifndef NSL_EXECUTABLE_BIAS_SHIFT_TPP
#define NSL_EXECUTABLE_BIAS_SHIFT_TPP

#include "NSL.hpp"
#include "complex.hpp"
#include "concepts.hpp"
#include <c10/core/TensorOptions.h>

//! Constant shift
/*!
 * This transformer takes a pre-defined constant shift and adds it to the input.
 *
 * \[
 * f(x) = x + shift
 * \]
 *
 * The inverse subtracts the shift
 * \[
 * f^{-1}(x) = x - shift
 * \]
 * */
template<NSL::Concept::isNumber Type>
struct ConstantShift : torch::nn::Module {
    ConstantShift(Type shift) :
        shift( shift )
    {
        
    }

    std::pair<torch::Tensor, Type> forward(torch::Tensor x) {
        return {x + shift, 0.};
    }

    std::pair<torch::Tensor, Type> inverse(torch::Tensor x) {
        return {x - shift, 0.};
    }

    // Use one of many "standard library" modules.
    NSL::complex<double> shift;
};


//! Shift
/*!
 * This transformer takes a pre-defined shift per input element and adds it to the input.
 *
 * \[
 *  f(x)_i = x_i + shift_i
 * \]
 *
 * The inverse subtracts the shift
 * \[
 * f^{-1}(x)_i = x_i - shift_i
 * \]
 * */
template<NSL::Concept::isNumber Type>
struct Shift : torch::nn::Module {
    Shift(NSL::RealTypeOf<Type> offset, torch::Tensor shift) :
        offset( offset ),
        shift( shift )
    {}

    template<NSL::Concept::isIndexer ... ShapeType>
    Shift( NSL::RealTypeOf<Type> offset, ShapeType ... shape) :
        offset( offset )
    {
        shift = register_parameter(
            "shift", 
            torch::zeros({shape...}, 
                torch::TensorOptions()
                    .template dtype<NSL::RealTypeOf<Type>>()
                    .requires_grad(true)
            )
        );

        // initialize using xavier uniform
        auto boundary = 1./std::sqrt(static_cast<double>(shift.numel()));

        torch::nn::init::uniform_(
            shift, 
            -boundary, boundary
        );
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        return {x + I*(offset + shift), torch::zeros({x.size(0)}, x.options()) };
    }

    std::pair<torch::Tensor, torch::Tensor> inverse(torch::Tensor x) {
        return {x - I*(offset + shift), torch::zeros({x.size(0)}, x.options()) };
    }

    torch::Scalar offset;
    torch::Tensor shift;
    const c10::complex<NSL::RealTypeOf<Type>> I{0,1};
};

#endif // NSL_EXECUTABLE_BIAS_SHIFT_TPP
