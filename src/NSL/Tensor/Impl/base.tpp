#ifndef NSL_TENSOR_IMPL_BASE_TPP
#define NSL_TENSOR_IMPL_BASE_TPP

#include "../../assert.hpp"
#include "../../concepts.hpp"

namespace NSL{
// declare the interface as many operators will need to return a Tensor 
// type.
template <NSL::Concept::isNumber Type> class Tensor;
} // namespace NSL

namespace NSL::TensorImpl {

template <NSL::Concept::isNumber Type>
class TensorBase {
    public:
    //! Real Type of a NSL::complex
    /*!
     * For Tensors of complex type RealType is provided as a convenience member.
     * It is accessible as Tensor<NSL::complex<RealType>>::RealType
     * If the Tensor is real valued `RealType = Type` as of the implementation of
     * `NSL::RT_extractor`
     */
     using RealType = typename NSL::RT_extractor<Type>::value_type;

    //! default constructor
    TensorBase() :
        data_(torch::zeros({}, torch::TensorOptions().dtype<Type>()) )
    {
        //std::cout << "NSL::Tensor()" << std::endl;
    }

    //! D-dimensional constructor
    template<NSL::Concept::isIntegral ... SizeTypes>
    constexpr explicit TensorBase(const NSL::size_t & size0, const SizeTypes & ... sizes) :
            data_(torch::zeros({size0,sizes...},
                               torch::TensorOptions().dtype<Type>()
            ))
    {
        //std::cout << "NSL::Tensor(const SizeTypes & ...)" << std::endl;
    }

    //! copy constructor
    /*!
     * Copy the data from other into this new instance. 
     * The copy constructor creates a shallow copy, i.e. the two tensors
     * share the underlying data.
     * For a deep copy consider the assignment operator `operator=`
     * */
    constexpr TensorBase(const NSL::TensorImpl::TensorBase<Type>& other):
            data_(other.data_)
    {
        // std::cout << "NSL::Tensor(const NSL::Tensor &)" << std::endl;
    }

    //! move constructor
    constexpr TensorBase(NSL::TensorImpl::TensorBase<Type> && other):
        data_(std::move(other.data_))
    {
        // std::cout << "NSL::Tensor(NSL::Tensor &&)" << std::endl;
    }

    //! move constructor
    constexpr TensorBase(const NSL::TensorImpl::TensorBase<Type> && other):
        data_(std::move(other.data_))
    {
        std::cout << "NSL::Tensor(const NSL::Tensor &&)" << std::endl;
    }

    //! type conversion
    template<NSL::Concept::isNumber OtherType>
    operator NSL::Tensor<OtherType> (){
        //! \todo: Is this efficient?
        return this->data_.to(torch::TensorOptions().dtype<OtherType>());
    }
    
    //! copy from `torch::Tensor` constructor
    constexpr TensorBase(const torch::Tensor & other) :
            data_(other)
    {
        //std::cout << "NSL::Tensor(const torch::Tensor & other)" << std::endl;
    }

    //! move from `torch::Tensor` constructor
    constexpr TensorBase(torch::Tensor && other) :
        data_(std::move(other))
    {
        //std::cout << "NSL::Tensor(torch::Tensor && other)" << std::endl;
    }

    //! conversion to `torch::Tensor` (creates a view on the underlying data)
    operator torch::Tensor(){
        return data_;
    }

    //! conversion to `const torch::Tensor` (creates a view on the underlying data)
    operator torch::Tensor() const {
        return data_;
    }

    template<NSL::Concept::isNumber PrintType>
    friend std::ostream & operator<<(std::ostream & os, const NSL::Tensor<PrintType> & tensor);

    protected:
    //! explicitly convert a polymorphism to this class by performing a shallow copy
    /*!
     * This is a convenience function such that other Impl classes can return
     * a NSL::Tensor<Type>. Nothing happens to the underlaying data and it 
     * contains the same address.
     *  \todo: Does the compile optimize this away?
     *  \todo: Is this publicly available in NSL::Tensor?
     * */
    explicit TensorBase<Type>(TensorBase<Type> * other) : 
        TensorBase(*other)
    {
        //std::cout << "NSL::Tensor(NSL::TensorBase *)" << std::endl;
    }

    //! Underlying torch::Tensor holding the data
    torch::Tensor data_;
    
    //! Translate D indices to linear index for memory access.
    template<NSL::Concept::isIntegral ... SizeTypes>
    NSL::size_t linearIndex_(const SizeTypes &... indices) const{
        // check that the number of arguments in indices matches the dimension of the tensor
        assertm(!(sizeof...(indices) < data_.dim()), "operator()(const SizeType &... indices) called with to little indices");
        assertm(!(sizeof...(indices) > data_.dim()), "operator()(const SizeType &... indices) called with to many indices");

        // unpack the parameter pack
        std::array<size_t, sizeof...(indices)> a_indices = {indices...};

        size_t offset = 0;
        for(size_t d = 0 ; d < sizeof...(indices); ++d){
            offset += a_indices[d] * data_.stride(d);
        }

        return offset;
    }



};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_BASE_TPP
