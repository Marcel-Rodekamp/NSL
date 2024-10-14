#ifndef NSL_CG_TPP
#define NSL_CG_TPP

#include "CG.hpp"

#include "../../../LinAlg/inner_product.tpp"
#include "../../../LinAlg/complex.tpp"
#include "../../../Tensor/Factory/like.tpp"
#include "complex.hpp"
#include "logger.hpp"
#include "IO/to_string.tpp"

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
void CG<Type>::optimize_for_GPU(const NSL::Tensor<Type> & b){
    if (b.device() == NSL::Device("cpu")){
        GPU_optimization_ = false;
        batchsize_ = 1;
        NSL::Logger::warn("Turning off GPU optimization for CG solver. The input vector is on CPU");
        return;
    }
    x_ = b;
    t_ = this->M_(x_);
    r_ = b-t_;
    rsqr_curr_ = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    rsqr_prev_ = rsqr_curr_;
    p_ = r_;
    auto warmupStream = at::cuda::getStreamFromPool();
    auto captureStream = at::cuda::getStreamFromPool();
    auto legacyStream = at::cuda::getCurrentCUDAStream();

    at::cuda::setCurrentCUDAStream(warmupStream);

    stream_sync(legacyStream, warmupStream);
    PUSH_RANGE("Graph Warmup",0);
    for (int iter = 0; iter < 10; iter++) {
        PUSH_RANGE("Warmup-Iteration",1);
        CG_iteration_base_();
        POP_RANGE;
    }
    POP_RANGE;

    stream_sync(warmupStream, captureStream);
    at::cuda::setCurrentCUDAStream(captureStream);
    PUSH_RANGE("Graph Capture (batch)",0);
    graph_.capture_begin();
    for (NSL::size_t i = 0; i < batchsize_; i++){
        CG_iteration_base_();
    }
    graph_.capture_end();
    POP_RANGE;

    stream_sync(captureStream, legacyStream);
    at::cuda::setCurrentCUDAStream(legacyStream);
    NSL::Logger::info("CG Graph captured");
}

template<NSL::Concept::isNumber Type >
void CG<Type>::CG_iteration_base_(){
    // compute the matrix vector product to determine the direction
    // t = M @ p
    t_ = this->M_(p_);
    alpha_ = rsqr_prev_ / (NSL::LinAlg::conj(p_) * t_).tensor_sum();
    x_ += alpha_ * p_;
    r_ -= alpha_ * t_;
    rsqr_curr_ = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    beta_ = rsqr_curr_ / rsqr_prev_;
    p_ = r_ + beta_ * p_;
    rsqr_prev_ = rsqr_curr_;
}

template<NSL::Concept::isNumber Type >
void CG<Type>::CG_iteration_(){
    if (GPU_optimization_){
        graph_.replay();
    } else {
        CG_iteration_base_();
    }
}

template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b ){
    return (*this)(b, b);
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

    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    rsqr_curr_ = NSL::real( (NSL::LinAlg::conj(r_) * r_).tensor_sum() );
    rsqr_prev_ = rsqr_curr_;

    // if the guess is already good enough return
    auto rsqr_curr_cpu = rsqr_curr_.to(NSL::CPU());
    if (rsqr_curr_cpu[0] <= errSq_) {
        NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),0);
        return x_;
    }

    // The initial gradient vector is then given by the residual
    p_ = r_;

    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; count+=batchsize_){
        PUSH_RANGE("CG Iteration",6);
        CG_iteration_();

        // now prepare the previous residual square for the next iteration
        // rsqr_prev = rsqr_curr;
        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        rsqr_curr_cpu = rsqr_curr_.to(NSL::CPU());
        if (rsqr_curr_cpu[0] <= errSq_) {
            NSL::Logger::debug("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr_cpu[0]),NSL::LinAlg::sqrt(errSq_),count);
            POP_RANGE;
            POP_RANGE;
            return x_;
        }

        // On debug level we print the solver status every step
        NSL::Logger::debug("CG Iteration: {}/{} | α = {} | ε² = {} |  β = {}", count, maxIter_, NSL::to_string(alpha_.to(NSL::CPU())[0]), rsqr_curr_cpu[0], beta_.to(NSL::CPU())[0]);
        POP_RANGE;
    } // for(counter)

    NSL::Logger::error("Error CG did not converge within {} iterations! |r| = {}", maxIter_, NSL::LinAlg::sqrt(rsqr_prev_.to(NSL::CPU())[0]));

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // operator()

} // namespace NSL::LinAlg

#undef USE_NVTX
#endif //NSL_CG_TPP