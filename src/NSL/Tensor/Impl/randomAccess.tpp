#ifndef NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP
#define NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP

#include "base.tpp"
#include "../../slice.hpp"

namespace NSL::TensorImpl {

template <NSL::Concept::isNumber Type>
class TensorRandomAccess:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Random Access Operator
    /*!
    *  \param indices: `NSL::size_t`, Indices for each dimension of the tensor
    *
    * Returns the element of the tensor associated with the indices.
    *
    **/
    template<NSL::Concept::isIntegral... SizeTypes >
    Type & operator()(const NSL::size_t & index0, const SizeTypes & ... indices){
        return this->data_.template data_ptr<Type>()[this->linearIndex_(index0,indices...)];
    }

    //! Random Access Operator (const)
    /*!
     *  \param indices: `NSL::size_t`, Indices for each dimension of the tensor
     *
     * Returns the element of the tensor associated with the indices.
     *
     * */
    template<NSL::Concept::isIntegral... SizeTypes >
    const Type & operator()(const NSL::size_t & index0,const SizeTypes & ... indices) const {
        return this->data_.template data_ptr<Type>()[this->linearIndex_(index0,indices...)];
    }

    //! Random Slice Access Operator
    /*!
     *  \param slices: `NSL::Slice`, Each argument slices the corresponding 
     *                              tensor dimension according to the 
     *                              start,stop,step arguments of the Slice 
     *                              objekt.
     *
     *  The returned tensor is of reduced size but stores the same underlying 
     *  data. Consequently, any change to the sliced tensor changes the original
     *  tensor.
     *
     * */
    template<NSL::Concept::isType<NSL::Slice> ... SliceTypes>
    NSL::Tensor<Type> operator()(NSL::Slice slice0, SliceTypes ... slices){
        return std::move(this->data_.index(std::initializer_list<torch::indexing::TensorIndex>{torch::indexing::Slice(slice0),torch::indexing::Slice(slices)...}));
    }

    //! Random Slice Access Operator (const)
    /*!
     *  \param slices: `NSL::Slice`, Each argument slices the corresponding 
     *                              tensor dimension according to the 
     *                              start,stop,step arguments of the Slice 
     *                              objekt.
     *
     *  The returned tensor is of reduced size but stores the same underlying 
     *  data. Consequently, any change to the sliced tensor changes the original
     *  tensor.
     *
     * */
    template<NSL::Concept::isType<NSL::Slice> ... SliceTypes>
    const NSL::Tensor<Type> operator()(NSL::Slice slice0, SliceTypes ... slices) const {
        return std::move(this->data_.index(std::initializer_list<torch::indexing::TensorIndex>{torch::indexing::Slice(slice0),torch::indexing::Slice(slices)...}));
    }

    //! Linear Random Access Operator
    /*!
     *  \param index0 Index of the first dimension
     *  \param indices: Indices of subsequent dimensions
     *
     * Returns the element of the tensor associated with the linearized index.
     * The (default) memory layout is
     * \f[
     *      i = \sum_{d=0}^{D-1} \mathcal{S}_d \cdot i_d
     * \f]
     * with given strides/offsets \f( \mathcal{S}_d \f) per dimension \f(d = 0,1,\dots D-1\f)
     *
     * */
    template<NSL::Concept::isIntegral SizeType>
    inline Type & operator[](const SizeType & index){
        return this->data_.template data_ptr<Type>()[index];
    }

    //! Const Linear Random Access Operator
    /*!
     *  \param index0 Index of the first dimension
     *  \param indices: Indices of subsequent dimensions
     *
     * Returns the element of the tensor associated with the linearized index.
     * The (default) memory layout is
     * \f[
     *      i = \sum_{d=0}^{D-1} \mathcal{S}_d \cdot i_d
     * \f]
     * with given strides/offsets \f( \mathcal{S}_d \f) per dimension \f(d = 0,1,\dots D-1\f)
     *
     * */
    template<NSL::Concept::isIntegral SizeType>
    const Type & operator[](const SizeType & index) const {
        return this->data_.template data_ptr<Type>()[index];
    }

    //! Pointer Access
    /*!
     * Access the underlying (C) pointer of the Tensor. Notice if the Tensor is
     * on a device this pointer will be a device pointer!
     * */
    Type * data(){
        return this->data_.template data_ptr<Type>();
    }

    //! Pointer Access (const)
    /*!
     * Access the underlying (C) pointer of the Tensor. Notice if the Tensor is
     * on a device this pointer will be a device pointer!
     * */
    const Type * data() const {
        return this->data_.template data_ptr<Type>();
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP
