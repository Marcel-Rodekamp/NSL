#include "gpu.hpp"

//=======================================================================
// GPU implementations
//=======================================================================
torch::DeviceType NSL::DEVICE::GPU::device() {
    return torch::kCUDA;
}

void NSL::DEVICE::GPU::synchronize(){
    torch::cuda::synchronize(this->deviceID);
}

NSL::DEVICE::Stream NSL::DEVICE::GPU::Stream(){
    return at::cuda::getStreamFromPool(/*isHighPriority=*/false, this->deviceID);
}

NSL::DEVICE::Stream NSL::DEVICE::GPU::DefaultStream(){
    return at::cuda::getDefaultCUDAStream(this->deviceID);
}

NSL::DEVICE::Stream NSL::DEVICE::GPU::CurrentStream(){
    return at::cuda::getCurrentCUDAStream(this->deviceID);
}

void NSL::DEVICE::GPU::CurrentStream(NSL::DEVICE::Stream stream){
    at::cuda::setCurrentCUDAStream(stream);
}

NSL::DEVICE::StreamGuard NSL::DEVICE::GPU::StreamGuard(){
    return at::cuda::CUDAStreamGuard(this->CurrentStream());
}

NSL::DEVICE::StreamGuard StreamGuard(NSL::DEVICE::Stream stream){
        return at::cuda::CUDAStreamGuard(stream);
}


//=======================================================================
// CPU implementations
//=======================================================================
torch::DeviceType NSL::DEVICE::CPU::device() {
    return torch::kCPU;
}
