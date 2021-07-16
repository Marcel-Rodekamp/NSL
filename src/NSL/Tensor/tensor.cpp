#include "tensor.hpp"

namespace NSL {
template<>
Tensor<float>::Tensor(const std::vector<long int> & dims) {
    data_ = torch::zeros(dims,
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(torch::kStrided)
                        .device(torch::kCPU)
                        .requires_grad(false)
    );
}

template<>
Tensor<double>::Tensor(const std::vector<long int> & dims){
    data_ = torch::zeros(dims,
                   torch::TensorOptions()
                       .dtype(torch::kFloat64)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false)

    );
}

} // namespace NSL