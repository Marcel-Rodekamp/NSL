#ifndef NSL_TENSOR_IMPL_DEVICE_TPP
#define NSL_TENSOR_IMPL_DEVICE_TPP

#include "base.tpp"
#include "device.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorDevice:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Copy data to device
    /*!
     * \param device, NSL::Device instance, which is the target of the function
     *
     * This is non blocking call
     * */
    NSL::Tensor<Type> to(NSL::Device device){
        return this->data_.to(device.device().device());
    }

    //! Copy data to device
    /*!
     * \param device, NSL::Device instance, which is the target of the function
     * \param non_blocking, if True makes the copy call non blocking.
     *
     * This can be non blocking call
     * */
    NSL::Tensor<Type> to(NSL::Device device, bool non_blocking){
        return this->data_.to(device.device().device(),non_blocking);
    }

    //! Copy data to device (inplace)
    /*!
     * \param device, NSL::Device instance, which is the target of the function
     * \param non_blocking, if True makes the copy call non blocking.
     *
     * This can be non blocking call
     * */
    void to(bool inplace, NSL::Device device, bool non_blocking = false){
        this->data_ = this->data_.to(device.device().device(),non_blocking);
    }

    //! Get device information from the Tensor
    const NSL::Device device(){
        return NSL::Device::fromTorch(
            this->data_.device()
        );
    }

    //! Get device information from the Tensor
    const NSL::Device device() const {
        return NSL::Device::fromTorch(
            this->data_.device()
        );
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_DEVICE_TPP
