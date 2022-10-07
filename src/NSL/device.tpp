#ifndef NSL_DEVICE_TPP
#define NSL_DEVICE_TPP

#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <stdexcept>
#include <torch/torch.h>

#include "types.hpp"

namespace NSL {

//! Device Handle
/*!
 * This struct is meant to store/pass information about a device for a given
 * `NSL::Tensor`.
 * This means this struct holds information about 
 *  - Location of the data, i.e. CPU/GPU
 *  - Splitting of the data, i.e. split around multiple CPUs/GPUs
 * how the Tensor is spl
 * */
class Device {
    public:
    Device() = delete;
    Device(const Device &) = default;
    Device(Device &&) = default;

    //! Constructor
    /*!
     * \param deviceIdentifier, String that specifies the use of CPU or GPU
     *        Allowed Values:
     *        - "GPU"
     *        - "CPU"
     *
     * */
    Device(std::string deviceIdentifier) :
        dev_(torch::TensorOptions().device(
                (deviceIdentifier == "GPU") ? torch::kCUDA : torch::kCPU
            )
        ) 
    {}

    Device(std::string deviceIdentifier, const NSL::size_t ID) :
        dev_(torch::TensorOptions().device(
                (deviceIdentifier == "GPU") ? torch::kCUDA : torch::kCPU,
                ID
            )
        ) 
    {}

    torch::TensorOptions device() {
        return dev_;
    }
    
    private:
    //! This is an enum for the different devices
    /*!
     * It stores multiple information about the hardware achrchitecture 
     * used
     * 
     * For reference see:
     *  - https://pytorch.org/cppdocs/api/enum_namespacec10_1a815bc73d9ef8591e4a92a70311b71697.html#_CPPv4N3c1010DeviceTypeE
     *  - https://pytorch.org/cppdocs/notes/tensor_creation.html
     * */
    // we store this information immutable. A new Device object is needed to be created
    //const c10::Device device;
    const torch::TensorOptions dev_;

   //! ToDo: Figure out how we can access different cpus etc.
};

//! Device Hande: GPU
struct GPU : public Device{
    GPU() : Device("GPU") {}
    GPU(const NSL::size_t ID) : Device("GPU",ID) {}
};

//! Device Hande: CPU
struct CPU : public Device{
    CPU() : Device("CPU") {}
    CPU(const NSL::size_t ID) : Device("CPU",ID) {}
};

//! Checks if a gpu is available
bool gpu_available(){
    return torch::cuda::is_available();
}

} // namspace NSL

#endif //NSL_DEVICE_TPP
