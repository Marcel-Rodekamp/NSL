#ifndef NSL_CG_TPP
#define NSL_CG_TPP

#include "CG.hpp"

#include "../../../LinAlg/inner_product.tpp"
#include "../../../LinAlg/complex.tpp"
#include "../../../Tensor/Factory/like.tpp"
#include "complex.hpp"
#include "logger.hpp"
#include "IO/to_string.tpp"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#define USE_NVTX
#include "profiling.hpp"

void stream_sync(
    at::cuda::CUDAStream& dependency,
    at::cuda::CUDAStream& dependent) {
  at::cuda::CUDAEvent cuda_ev;
  cuda_ev.record(dependency);
  cuda_ev.block(dependent);
}

namespace NSL::LinAlg{

template<NSL::Concept::isNumber Type >
void CG<Type>::CG_iteration(NSL::Tensor<Type> & alpha, NSL::Tensor<typename NSL::RT_extractor<Type>::type> & rsqr_curr, NSL::Tensor<typename NSL::RT_extractor<Type>::type> & rsqr_prev, NSL::Tensor<typename NSL::RT_extractor<Type>::type> & beta){
    // compute the matrix vector product to determine the direction
    // t = M @ p
    t_ = this->M_(p_);

    alpha = rsqr_prev / (NSL::LinAlg::conj(p_) * t_).tensor_sum();
    x_ += alpha * p_;
    r_ -= alpha * t_;
    rsqr_curr = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    beta = rsqr_curr / rsqr_prev;
    p_ = r_ + beta * p_;
    rsqr_prev = rsqr_curr;
    
}

template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b ){
    // This algorithm can be found e.g.: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    //


    // initialize the solution vector x_ which after convergence 
    // stores the approximate result x = M^{-1} @ b.
    // Multiple initializations are possible and can enhance the convergence
    // see e.g. Preconditioning. Here we just choose a simple start vector
    // which is an arbitrary choise.
    x_ = b;

    // Compute the initial matrix vector product and store it in the 
    // corresponding vector t
    t_ = this->M_(b);

    // This initial matrix vector product defines the initial residual vector 
    r_ = b-t_;

    auto device = b.device();
    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> rsqr_curr(device, 1); 
    rsqr_curr = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> rsqr_prev(device, 1); 
    rsqr_prev = rsqr_curr;
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> beta(device, 1);
    NSL::Tensor<Type> alpha(device, 1);

    // if the guess is already good enough return
    auto rsqr_curr_cpu = rsqr_curr.to(NSL::CPU());
    if (rsqr_curr_cpu[0] <= errSq_) {
        NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),0);
        return x_;
    }

    // The initial gradient vector is then given by the residual
    p_ = r_;

    at::cuda::CUDAGraph graph;
    auto warmupStream = at::cuda::getStreamFromPool();
    auto captureStream = at::cuda::getStreamFromPool();
    auto legacyStream = at::cuda::getCurrentCUDAStream();

    at::cuda::setCurrentCUDAStream(warmupStream);

    stream_sync(legacyStream, warmupStream);

    for (int iter = 0; iter < 50; iter++) {
        PUSH_RANGE("Mp",0);
        CG_iteration(alpha, rsqr_curr, rsqr_prev, beta);
        POP_RANGE;
    }

    stream_sync(warmupStream, captureStream);
    at::cuda::setCurrentCUDAStream(captureStream);

    NSL::size_t batch_size = 100;
    graph.capture_begin();
    for (NSL::size_t i = 0; i < batch_size; i++){
        CG_iteration(alpha, rsqr_curr, rsqr_prev, beta);
    }
    graph.capture_end();

    stream_sync(captureStream, legacyStream);
    at::cuda::setCurrentCUDAStream(legacyStream);
    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; count+=batch_size){
        std::cout << "Iteration: " << count << std::endl;
        PUSH_RANGE("CG Iteration",6);
        PUSH_RANGE("Graph",0);
        graph.replay();
        POP_RANGE;
        // determine the scale of the orthogonalization
        //alpha{i} = (r{i},r{i})/(p{i},t{i}) (remember we stored (r{i},r{i}) in rsqr_prev)
        // alpha = rsqr_prev / (NSL::LinAlg::conj(p_) * t_).tensor_sum();

        // update the solution x according to the step
        // x{i+1} = x{i} + alpha{i} * p{i}
        // x_ += alpha * p_;

        // compute the residual 
        // r{i+1} = r{i} - alpha{i} * t{i}
        // r_ -= alpha * t_;

        // and the resulting error square
        // err = (r{i+1},r{i+1})
        // rsqr_curr = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );

        // compute the momentum update scale
        // beta{i} = (r{i+1},r{i+1})/(r{i},r{i}
        // beta = rsqr_curr / rsqr_prev;

        // update the momentum
        // p{i+1} = r{i+1} + beta{i} * p{i}
        // p_ = r_ + beta * p_;

        // now prepare the previous residual square for the next iteration
        // rsqr_prev = rsqr_curr;
        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        PUSH_RANGE("Condition and logging",1);
        rsqr_curr_cpu = rsqr_curr.to(NSL::CPU());
        if (rsqr_curr_cpu[0] <= errSq_) {
            NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),count);
            POP_RANGE;
            POP_RANGE;
            return x_;
        }

        // On debug level we print the solver status every step
        NSL::Logger::debug("CG Iteration: {}/{} | α = {} | ε² = {} |  β = {}", count, maxIter_, NSL::to_string(alpha.to(NSL::CPU())[0]), rsqr_curr_cpu[0], beta.to(NSL::CPU())[0]);
        POP_RANGE;
        POP_RANGE;
    } // for(counter)

    NSL::Logger::error("Error CG did not converge within {} iterations! |r| = {}", maxIter_, NSL::LinAlg::sqrt(rsqr_prev.to(NSL::CPU())[0]));

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // operator()

template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b, const NSL::Tensor<Type> & x0 ){
    // This algorithm can be found e.g.: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    //

    // initialize the solution vector x_ which after convergence 
    // stores the approximate result x = M^{-1} @ b.
    // Multiple initializations are possible and can enhance the convergence
    // see e.g. Preconditioning. Here we just choose a simple start vector
    // which is an arbitrary choise.
    x_ = x0;

    // Compute the initial matrix vector product and store it in the 
    // corresponding vector t
    t_ = this->M_(x_);

    // This initial matrix vector product defines the initial residual vector 
    r_ = b-t_;

    auto device = b.device();
    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> rsqr_curr(device, 1); 
    rsqr_curr = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> rsqr_prev(device, 1); 
    rsqr_prev = rsqr_curr;
    NSL::Tensor<typename NSL::RT_extractor<Type>::type> beta(device, 1);
    NSL::Tensor<Type> alpha(device, 1);

    // if the guess is already good enough return
    auto rsqr_curr_cpu = rsqr_curr.to(NSL::CPU());
    if (rsqr_curr_cpu[0] <= errSq_) {
        NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),0);
        return x_;
    }

    // The initial gradient vector is then given by the residual
    p_ = r_;

    at::cuda::CUDAGraph graph;
    auto warmupStream = at::cuda::getStreamFromPool();
    auto captureStream = at::cuda::getStreamFromPool();
    auto legacyStream = at::cuda::getCurrentCUDAStream();

    at::cuda::setCurrentCUDAStream(warmupStream);

    stream_sync(legacyStream, warmupStream);

    for (int iter = 0; iter < 50; iter++) {
        PUSH_RANGE("Mp",0);
        CG_iteration(alpha, rsqr_curr, rsqr_prev, beta);
        POP_RANGE;
    }

    stream_sync(warmupStream, captureStream);
    at::cuda::setCurrentCUDAStream(captureStream);

    NSL::size_t batch_size = 100;
    graph.capture_begin();
    for (NSL::size_t i = 0; i < batch_size; i++){
        CG_iteration(alpha, rsqr_curr, rsqr_prev, beta);
    }
    graph.capture_end();

    stream_sync(captureStream, legacyStream);
    at::cuda::setCurrentCUDAStream(legacyStream);
    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; count+=batch_size){
        std::cout << "Iteration: " << count << std::endl;
        PUSH_RANGE("CG Iteration",6);
        PUSH_RANGE("Graph",0);
        graph.replay();
        POP_RANGE;
        // determine the scale of the orthogonalization
        //alpha{i} = (r{i},r{i})/(p{i},t{i}) (remember we stored (r{i},r{i}) in rsqr_prev)
        // alpha = rsqr_prev / (NSL::LinAlg::conj(p_) * t_).tensor_sum();

        // update the solution x according to the step
        // x{i+1} = x{i} + alpha{i} * p{i}
        // x_ += alpha * p_;

        // compute the residual 
        // r{i+1} = r{i} - alpha{i} * t{i}
        // r_ -= alpha * t_;

        // and the resulting error square
        // err = (r{i+1},r{i+1})
        // rsqr_curr = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );

        // compute the momentum update scale
        // beta{i} = (r{i+1},r{i+1})/(r{i},r{i}
        // beta = rsqr_curr / rsqr_prev;

        // update the momentum
        // p{i+1} = r{i+1} + beta{i} * p{i}
        // p_ = r_ + beta * p_;

        // now prepare the previous residual square for the next iteration
        // rsqr_prev = rsqr_curr;
        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        PUSH_RANGE("Condition and logging",1);
        rsqr_curr_cpu = rsqr_curr.to(NSL::CPU());
        if (rsqr_curr_cpu[0] <= errSq_) {
            NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),count);
            POP_RANGE;
            POP_RANGE;
            return x_;
        }

        // On debug level we print the solver status every step
        NSL::Logger::debug("CG Iteration: {}/{} | α = {} | ε² = {} |  β = {}", count, maxIter_, NSL::to_string(alpha.to(NSL::CPU())[0]), rsqr_curr_cpu[0], beta.to(NSL::CPU())[0]);
        POP_RANGE;
        POP_RANGE;
    } // for(counter)

    NSL::Logger::error("Error CG did not converge within {} iterations! |r| = {}", maxIter_, NSL::LinAlg::sqrt(rsqr_prev.to(NSL::CPU())[0]));

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // operator()

} // namespace NSL::LinAlg

#undef USE_NVTX
#endif //NSL_CG_TPP
