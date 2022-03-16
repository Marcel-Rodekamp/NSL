#ifndef NSL_TENSOR_IMPL_CONTRACTION_TPP
#define NSL_TENSOR_IMPL_CONTRACTION_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorContraction:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Compute the tensor contraction
    /*!
     * params:
     *      -- other, Tensor to contract with
     *      -- dims, dimensions to contract
     *
     *  This function computes the tensor contraction
     *  $$
     *      v^{\mu_1,\mu_2,...\mu_{m-d}, ..., \nu_d,...,\nu_n} =
     *      \sum_{k_0} \sum_{k_1} ... \sum_{k_{d-1}}
     *      T1^{\mu_1,\mu_2,...,\mu_{m-d}, k_0, k_1, ..., k_{d-1}} \cdot
     *      T2^{ k_0, k_1, ..., k_{d-1}, \nu_d,...\nu_n }
     *  $$
     *
    */
    template<NSL::Concept::isIntegral ... TypesK>
    Tensor<Type> contract(const NSL::Tensor<Type> & other, const TypesK & ... ks){
        return torch::tensordot(this->data_, other.data_, {ks...}, {ks...} );
    }



};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_CONTRACTION_TPP
