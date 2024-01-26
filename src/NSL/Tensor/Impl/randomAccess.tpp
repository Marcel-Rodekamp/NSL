#ifndef NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP
#define NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP

#include "base.tpp"
#include "../../sliceObj.tpp"
#include "../../paramPack.hpp"
#include <type_traits>


namespace NSL::TensorImpl {

template <NSL::Concept::isNumber Type>
class TensorRandomAccess:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    //! Accessor for indexer Access (NSL::Slice, NSL::Ellipsis, ...) as well as int NSL::size_t convertibles
    template<NSL::Concept::isIndexer ... IndexTypes>
        requires(NSL::packContainsDerived<NSL::Indexer,IndexTypes...>::value  )
    NSL::Tensor<Type> operator()(IndexTypes ... indexer) {
        return this->data_.index(
                std::initializer_list<torch::indexing::TensorIndex>{
                    torch::indexing::TensorIndex(indexer)...
                }
        );
    }

    //! Accessor for pure NSL::size_t access
    template<NSL::Concept::isIndexer ... IndexTypes>
        requires(!NSL::packContainsDerived<NSL::Indexer,IndexTypes...>::value  )
    Type & operator()(IndexTypes ... indices) {
        return this->data_.template data_ptr<Type>()[this->linearIndex_(indices...)];
    }

    //! Accessor for indexer Access (NSL::Slice, NSL::Ellipsis, ...) as well as int NSL::size_t convertibles
    template<NSL::Concept::isIndexer ... IndexTypes>
        requires(NSL::packContainsDerived<NSL::Indexer,IndexTypes...>::value  )
    NSL::Tensor<Type> operator()(IndexTypes ... indexer) const {
        return this->data_.index(
                std::initializer_list<torch::indexing::TensorIndex>{
                    torch::indexing::TensorIndex(indexer)...
                }
        );
    }

    //! Accessor for pure NSL::size_t access
    template<NSL::Concept::isIndexer ... IndexTypes>
        requires(!NSL::packContainsDerived<NSL::Indexer,IndexTypes...>::value  )
    NSL::Tensor<Type> & operator()(IndexTypes ... indices) const {
        return this->data_.index(
                std::initializer_list<torch::indexing::TensorIndex>{
                    torch::indexing::TensorIndex(indices)...
                }
        );
        //return this->data_.template data_ptr<Type>()[this->linearIndex_(indices...)];
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

    template<NSL::Concept::isIndexer ... IndexTypes>
        requires(NSL::packContainsDerived<NSL::Indexer,IndexTypes...>::value  )
    NSL::Tensor<Type> index_fill(const Type & value, IndexTypes ... indices){
        this->data_.index_put_( 
                std::initializer_list<torch::indexing::TensorIndex>{
                    torch::indexing::TensorIndex(indices)...
                },
                value
        );
        return *this;
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_RANDOM_ACCESS_TPP
