#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

//! \file tensor.hpp

#include "../complex.hpp" // get NSL::RT_extractor
#include "../concepts.hpp" // get NSL::Concept::


// Include the implementations of Tensor
#include "Impl/base.tpp"
#include "Tensor/Impl/base.tpp"
#include "Tensor/Impl/factory.tpp"
#include "Tensor/Impl/print.tpp"
#include "Tensor/Impl/randomAccess.tpp"
#include "Tensor/Impl/slice.tpp"
#include "Tensor/Impl/stats.tpp"
#include "Tensor/Impl/realImag.tpp"
#include "Tensor/Impl/expand.tpp"
#include "Tensor/Impl/shift.tpp"

// Arithmetic
#include "Tensor/Impl/operatorAdditionEqual.tpp"
#include "Tensor/Impl/operatorSubtractionEqual.tpp"
#include "Tensor/Impl/operatorMultiplicationEqual.tpp"
#include "Tensor/Impl/operatorDivisionEqual.tpp"

// Linear Algebra 
#include "Tensor/Impl/transpose.tpp"
#include "Tensor/Impl/complexConjugate.tpp"
#include "Tensor/Impl/adjoint.tpp"
#include "Tensor/Impl/abs.tpp"
#include "Tensor/Impl/contraction.tpp"
#include "Tensor/Impl/matrixExp.tpp"
#include "Tensor/Impl/trigonometric.tpp"
#include "Tensor/Impl/reductions.tpp"

namespace NSL{

//! Tensor class holding d-dimensional data and providing various algebraic methods.
template <NSL::Concept::isNumber Type>
class Tensor:
    virtual public NSL::TensorImpl::TensorBase<Type>,
    public NSL::TensorImpl::TensorFactories<Type>,
    public NSL::TensorImpl::TensorRandomAccess<Type>,
    public NSL::TensorImpl::TensorAdditionEqual<Type>,
    public NSL::TensorImpl::TensorSubtractionEqual<Type>,
    public NSL::TensorImpl::TensorMultiplicationEqual<Type>,
    public NSL::TensorImpl::TensorDivisionEqual<Type>,
    public NSL::TensorImpl::TensorSlice<Type>,
    public NSL::TensorImpl::TensorStats<Type>,
    public NSL::TensorImpl::TensorTranspose<Type>,
    public NSL::TensorImpl::TensorComplexConj<Type>,
    public NSL::TensorImpl::TensorAdjoint<Type>,
    public NSL::TensorImpl::TensorReal<Type>,
    public NSL::TensorImpl::TensorImag<Type>,
    public NSL::TensorImpl::TensorAbs<Type>,
    public NSL::TensorImpl::TensorContraction<Type>,
    public NSL::TensorImpl::TensorMatrixExp<Type>,
    public NSL::TensorImpl::TensorTrigonometric<Type>,
    public NSL::TensorImpl::TensorReductions<Type>,
    public NSL::TensorImpl::TensorExpand<Type>,
    public NSL::TensorImpl::TensorShift<Type>
{
    public:

    Tensor() = default;
    Tensor(Tensor &&) = default;
    Tensor(const Tensor &) = default;

    // import the constructors
    using NSL::TensorImpl::TensorBase<Type>::TensorBase;
    //! \todo: Inherit documentation of constructors, INLINE_INHERITED_MEMB omits those.
    

    // For some reason I can't export the assignment operators so they
    // need to come here. Other members are and their definitions are 
    // listed below
    //! assignment operator with value
    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<Type> & operator=(const OtherType & other){
        this->data_.fill_(other);
        return *this;
    }

    //! assignement operator
    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<Type> & operator=(const NSL::Tensor<OtherType> & other){
        // deep copy of other into this
        // by default GPU <-> is asynch on host site
        this->data_.copy_(other,true);
        return *this;
    }

    //! assignement operator
    NSL::Tensor<Type> & operator=(const NSL::Tensor<Type> & other){
        // deep copy of other into this
        // by default GPU <-> is asynch on host site
        this->data_.copy_(other,true);
        return *this;
    }

    //! assignement operator from torch
    NSL::Tensor<Type> & operator=(const torch::Tensor & other){
        // deep copy of other into this
        // by default GPU <-> is asynch on host site
        this->data_.copy_(other,true);
        return *this;
    }

    /* This is the interface class users should be using.
     * All the different implementations are found in the files in Impl/ .
     * The following lists provides an overview where to find what:
     *
     * - Base class: Impl/base.tpp
     *      - Underlaying data implementation/interface to torch:Tensor (protected)
     *      - Linear Indexing (protected)
     *      - Constructors
     *      - Conversion from and to torch
     *      - operator<< (implemented in print.tpp)
     * - Factories: Impl/factory.tpp
     *      - Random data fill e.g. Tensor.rand()
     * - Random Access: Impl/randomAccess.tpp
     *      - random access operators: operator(NSL::size_t ...) & operator(NSL::Slice ...)
     *      - Pointer access: data()
     * - Equality: Impl/operatorEqual.tpp
     *      - operator==
     * - Not Equality: Impl/operatorNotEqual.tpp
     *      - operator!=
     * - Greater: Impl/operatorGreater.tpp
     *      - operatpr>
     * - Greater Equal: Impl/operatorGreaterEqual.tpp
     *      - operatpr>=
     * - Smaller: Impl/operatorSmaller.tpp
     *      - operatpr<
     * - Smaller Equal: Impl/operatorSmallerEqual.tpp
     *      - operator<=
     * - Addition: Impl/operatorAddition.tpp
     *      - operator+
      * - Addition Equal: Impl/operatorAdditionEqual.tpp
     *      - operator+=
     * - Subtraction: Impl/operatorsubtraction.tpp
     *      - operator-
     * - Subtraction Equal: Impl/operatorSubtractionEqual.tpp
     *      - operator-=
     * - Multiplication: Impl/operatorMultiplication.tpp
     *      - operator*
     * - Multiplication Equal: Impl/operatorMultiplicationEqual.tpp
     *      - operator*=
     * - Division: Impl/operatorDivision.tpp
     *      - operator/
     * - Division Equal: Impl/operatorDivisionEqual.tpp
     *      - operator/=
     * - Slice: Impl/slice.tpp
     *      - slice
     * - Stats: Impl/stats.tpp
     *      - shape
     *      - dim
     *      - numel
     * - Transpose: Impl/transpose.tpp
     *      - transpose
     *      - T
     * - Adjoint: Impl/adjoint.tpp
     *      - adjoint
     *      - H
     * - Real & Imag part: Impl/realImag.tpp
     *      - real
     *      -imag
     * - abs: Impl/abs.tpp
     *      - abs
     * - Tensor contraction: Impl/contraction.tpp
     *      - contraction
     * - Matrix exponential: Impl/matrixExp.tpp
     *      - mat_exp
     * - Trigonometric functions: Impl/trigonometric.tpp
     *      - exp
     *      - sin
     *      - cos
     *      - tan
     *      - sinh
     *      - cosh
     *      - tanh
     * - Reductions: Impl/reductions.tpp
     *      - sum (+)
     *      - prod (*)
     *      - all (&&)
     *      - any (||)
     * - Expand: Impl/expand.tpp 
     *      - expand  
     * - Shift: Impl/shift.tpp
     *      - shift
     * */

};

}

// include externely defined operators 

// boolean
#include "Tensor/Impl/operatorEqual.tpp"
#include "Tensor/Impl/operatorNotEqual.tpp"
#include "Tensor/Impl/operatorGreater.tpp"
#include "Tensor/Impl/operatorGreaterEqual.tpp"
#include "Tensor/Impl/operatorSmaller.tpp"
#include "Tensor/Impl/operatorSmallerEqual.tpp"

// arithmetic
#include "Tensor/Impl/operatorAddition.tpp"
#include "Tensor/Impl/operatorSubtraction.tpp"
#include "Tensor/Impl/operatorMultiplication.tpp"
#include "Tensor/Impl/operatorDivision.tpp"

#endif //NSL_TENSOR_HPP
