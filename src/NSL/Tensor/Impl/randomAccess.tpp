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

//    //! Random Access operator
//    /*!
//     * \param indexer, pack of `NSL::Slice` and `NSL::size_t` types. 
//     * 
//     * This indexer mixes the usage of `NSL::Slice` and `NSL::size_t`
//     * to have a simpler access on a particular slice of a axis.
//     * */
//    template<NSL::Concept::isIndexer... IndexTypes>
//        // use only if NSL::size_t in IndexTypes and NSL::Slice in IndexTypes
//        requires (NSL::packContains<NSL::Indexer,IndexTypes...>::value && NSL::packContainsConvertible<NSL::size_t,IndexTypes...>::value)
//    NSL::Tensor<Type> operator()(IndexTypes ... indexer) 
//    {
//        return this->data_.index(
//                std::initializer_list<torch::indexing::TensorIndex>{
//                    torch::indexing::TensorIndex(indexer)...
//                }
//            );
//    }
//
//    //! Random Access operator (const)
//    /*!
//     * \param indexer, pack of `NSL::Slice` and `NSL::size_t` types. 
//     * 
//     * This indexer mixes the usage of `NSL::Slice` and `NSL::size_t`
//     * to have a simpler access on a particular slice of a axis.
//     * */
//    template<NSL::Concept::isIndexer... IndexTypes>
//        // use only if NSL::size_t in IndexTypes and NSL::Slice in IndexTypes
//        requires (NSL::packContains<NSL::Slice,IndexTypes...>::value && NSL::packContainsConvertible<NSL::size_t,IndexTypes...>::value)
//    NSL::Tensor<Type> operator()(IndexTypes ... indexer) const
//    {
//        return this->data_.index(
//                std::initializer_list<torch::indexing::TensorIndex>{
//                    torch::indexing::TensorIndex(indexer)...
//                }
//            );
//    }
//    
//    
//    //! Random Access Operator
//    /*!
//    *  \param indices: `NSL::size_t`, Indices for each dimension of the tensor
//    *
//    * Returns the element of the tensor associated with the indices.
//    *
//    **/
//    // hotfix: The require-clause is needed as the assignment construction
//    //         would be called if a mixed indexing is used with an integer
//    //         in the first place
//    template<NSL::Concept::isIntegral... SizeTypes >
//        requires (!NSL::packContains<NSL::Slice,SizeTypes...>::value)
//    Type & operator()(NSL::size_t index0, const SizeTypes & ... indices){
//        return this->data_.template data_ptr<Type>()[this->linearIndex_(index0,indices...)];
//    }
//
//    //! Random Access Operator (const)
//    /*!
//     *  \param indices: `NSL::size_t`, Indices for each dimension of the tensor
//     *
//     * Returns the element of the tensor associated with the indices.
//     *
//     * */
//    // hotfix: The require-clause is needed as the assignment construction
//    //         would be called if a mixed indexing is used with an integer
//    //         in the first place
//    template<NSL::Concept::isIntegral... SizeTypes >
//        requires (!NSL::packContains<NSL::Slice,SizeTypes...>::value)
//    const Type & operator()(NSL::size_t index0,const SizeTypes & ... indices) const {
//        return this->data_.template data_ptr<Type>()[this->linearIndex_(index0,indices...)];
//    }
//
//    //! Random Slice Access Operator
//    /*!
//     *  \param slices: `NSL::Slice`, Each argument slices the corresponding 
//     *                              tensor dimension according to the 
//     *                              start,stop,step arguments of the Slice 
//     *                              objekt.
//     *
//     *  The returned tensor is of reduced size but stores the same underlying 
//     *  data. Consequently, any change to the sliced tensor changes the original
//     *  tensor.
//     *
//     * */
//    // hotfix: The require-clause is needed as NSL::Slice can be constructed
//    //         from a single NSL::size_t. 
//    template<NSL::Concept::isType<NSL::Slice> ... SliceTypes>
//        requires (!NSL::packContainsConvertible<NSL::size_t,SliceTypes...>::value)
//    NSL::Tensor<Type> operator()(NSL::Slice slice0, SliceTypes ... slices){
//        return std::move(this->data_.index(std::initializer_list<torch::indexing::TensorIndex>{torch::indexing::Slice(slice0),torch::indexing::Slice(slices)...}));
//
//    }
//
//    //! Random Slice Access Operator (const)
//    /*!
//     *  \param slices: `NSL::Slice`, Each argument slices the corresponding 
//     *                              tensor dimension according to the 
//     *                              start,stop,step arguments of the Slice 
//     *                              objekt.
//     *
//     *  The returned tensor is of reduced size but stores the same underlying 
//     *  data. Consequently, any change to the sliced tensor changes the original
//     *  tensor.
//     *
//     * */
//    // hotfix: The require-clause is needed as NSL::Slice can be constructed
//    //         from a single NSL::size_t. 
//    template<NSL::Concept::isType<NSL::Slice> ... SliceTypes>
//        requires (!NSL::packContainsConvertible<NSL::size_t,SliceTypes...>::value)
//    const NSL::Tensor<Type> operator()(NSL::Slice slice0, SliceTypes ... slices) const {
//        return std::move(this->data_.index(std::initializer_list<torch::indexing::TensorIndex>{torch::indexing::Slice(slice0),torch::indexing::Slice(slices)...}));
//
//    }

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

    //! Accessor for pute NSL::size_t access
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

    /*
    //! Accessor for pute NSL::size_t access
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
    */

    //! Accessor for mixed indexer and int access

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
