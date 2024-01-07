#ifndef NSL_TORCH_TRANSFORMER_TPP
#define NSL_TORCH_TRANSFORMER_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "transformer.hpp"

namespace NSL{

template<
    Concept::isNumber Type, 
    NSL::Concept::isDerived<torch::nn::Module> TransformerType
>
class TorchTransformer: public Transformer<Type>{

public:

    TorchTransformer(TransformerType transformer): 
        torchTransformer_(transformer)
    {}

    //! A function that takes an input configuration applies the transformation and returns 
    //! the result and the log determinant of the associated Jacobian matrix.
    std::pair<NSL::Tensor<Type>, Type> forward(const NSL::Tensor<Type> & input ) override {
        auto [output, logDet] = torchTransformer_.forward(torch::Tensor(input));
        return std::make_pair<NSL::Tensor<Type>,Type>(output, std::move(logDet));
    }


    //! A function that takes an input configuration applies the transformation and returns 
    //! the result and the log determinant of the associated Jacobian matrix.
    std::pair<NSL::Configuration<Type>, Type> forward(NSL::Configuration<Type> & input ) override {

        NSL::Configuration<Type> output;
        Type logDetJ = 0;

        for(auto [key, inputTensor]: input){
            auto [outputTensor, logDet] = torchTransformer_.forward(torch::Tensor(inputTensor));
            output[key] = NSL::Tensor<Type> (outputTensor);
            logDetJ += logDet;
        }

        return std::make_pair<NSL::Configuration<Type>,Type>(
            std::move(output), std::move(logDetJ)
        );
    }

    //! A function that takes an input configuration applies the transformation and returns 
    //! the result and the log determinant of the associated Jacobian matrix.
    std::pair<NSL::Tensor<Type>, Type> inverse(const NSL::Tensor<Type> & input ) override {
        auto [output, logDet] = torchTransformer_.inverse(torch::Tensor(input));
        return std::make_pair<NSL::Tensor<Type>,Type>(output, std::move(logDet));
    }


    //! A function that takes an input configuration applies the transformation and returns 
    //! the result and the log determinant of the associated Jacobian matrix.
    std::pair<NSL::Configuration<Type>, Type> inverse(NSL::Configuration<Type> & input ) override {

        NSL::Configuration<Type> output;
        Type logDetJ = 0;

        for(auto [key, inputTensor]: input){
            auto [outputTensor, logDet] = torchTransformer_.inverse(torch::Tensor(inputTensor));
            output[key] = NSL::Tensor<Type> (outputTensor);
            logDetJ += logDet;
        }

        return std::make_pair<NSL::Configuration<Type>,Type>(
            std::move(output), std::move(logDetJ)
        );
    }

private:

    TransformerType torchTransformer_;

}; // torchTransformer

} // namespace NSL

#endif // NSL_TORCH_TRANSFORMER_TPP
