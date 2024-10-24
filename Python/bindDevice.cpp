#include <torch/torch.h>
#include <torch/extension.h> 

namespace pybind11 {
namespace detail {

template <>
struct type_caster<NSL::Device> {
    PYBIND11_TYPE_CASTER(NSL::Device, _("NSL::Device"));

    // Conversion from Python to C++
    bool load(handle src, bool) {
        torch::Device py_device = src.cast<torch::Device>();
        if (!py_device.is_cuda() && !py_device.is_cpu()) {
            throw std::runtime_error("Device must be either CPU or CUDA");
        }
        std::string device_identifier = py_device.is_cuda() ? "GPU" : "CPU";
        value = NSL::Device(device_identifier, py_device.index());
        return true;
    }

    // Conversion from C++ to Python
    static handle cast(const NSL::Device& src, return_value_policy policy, handle parent) {
        return py::cast(src.device(), policy, parent);
    }
};

} // namespace detail
} // namespace pybind11