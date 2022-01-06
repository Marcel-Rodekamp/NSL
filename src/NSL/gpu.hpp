#ifndef NANOSYSTEMLIBRARY_GPU_HPP
#define NANOSYSTEMLIBRARY_GPU_HPP

#include <torch/torch.h>

// at::cuda::CUDAStream
#include <c10/cuda/CUDAStream.h>
// at::cuda::CUDAStreamGuard
#include <c10/cuda/CUDAGuard.h>

namespace NSL {
namespace DEVICE {
//!
/*! GPU - stream
 * Multiple kernels can be executed asynchronously on the device. This is achievable
 * by using streams
 * \todo: Add some links here
 * PyTorch encapsulates the stream as `at::cuda::CUDAStream`. We abbreviate these
 * as Stream.
 * */
using Stream = at::cuda::CUDAStream;

using StreamGuard = at::cuda::CUDAStreamGuard;

//! Device object: Graphics Processing Unit
/*!
 * Construct a GPU object to maintain various information
 * */
struct GPU{
    int deviceID;

    constexpr explicit GPU(): deviceID(0) {}
    constexpr explicit GPU(const int deviceId): deviceID(deviceID) {};

    //! Check if any GPU is available
    static bool is_available(){
        return torch::cuda::is_available();
    }

    //! Returns the device (GPU + id)
    /*!
     *  This function is mainly ment for internal usage to communicate the devices.
     *  with the libtorch interface
     * */
    torch::DeviceType device();

    //! Synchronize the GPU with the CPU
    /*!
     *  This function synchronizes the GPU with the GPU. Note that all calls with
     *  GPU computation are asynchronous.
     * */
    void synchronize();

    //! Get new GPU-Stream from this device.
    /*!
     * \todo: Add Documentation
     * */
    NSL::DEVICE::Stream Stream();

    //! Get new default GPU-Stream from this device.
    /*!
     * \todo: Add Documentation
     * */
    NSL::DEVICE::Stream DefaultStream();

    //! Get current GPU-stream from this device.
    /*!
     * \todo: Add Documentation
     * */
    NSL::DEVICE::Stream CurrentStream();

    //! Set current GPU-stream for this device
    /*!
     * \todo: Add Documentation
     * */
    void CurrentStream(at::cuda::CUDAStream stream);

    //! Create a StreamGuard for the current stream of this device
    /*!
     * \todo: Add Documentation
     * */
    NSL::DEVICE::StreamGuard StreamGuard();


    //! Create a StreamGuard for the given stream
    /*!
     * \todo: Add Documentation
     * */
    NSL::DEVICE::StreamGuard StreamGuard(NSL::DEVICE::Stream stream);

};

struct CPU{
    int deviceID;

    constexpr explicit CPU(): deviceID(0) {}
    constexpr explicit CPU(const int deviceId): deviceID(deviceID) {};

    //! Returns the device (CPU + id)
    /*!
     *  This function is mainly ment for internal usage to communicate the devices.
     *  with the libtorch interface
     * */
    torch::DeviceType device();


};

} // namespace DEVICE

void synchronize();

} // namespace NSL

#endif //NANOSYSTEMLIBRARY_GPU_HPP
