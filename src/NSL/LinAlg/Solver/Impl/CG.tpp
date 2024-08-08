#ifndef NSL_CG_TPP
#define NSL_CG_TPP

#include "CG.hpp"

#include "../../../LinAlg/inner_product.tpp"
#include "../../../LinAlg/complex.tpp"
#include "../../../Tensor/Factory/like.tpp"
#include "complex.hpp"
#include "logger.hpp"
#include "IO/to_string.tpp"

namespace NSL::LinAlg{
template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::solve_(const NSL::Tensor<Type> & b ){
    // This algorithm can be found e.g.: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    //

    // Compute the initial matrix vector product and store it in the 
    // corresponding vector t
    t_ = this->Mfree_(x_);

    // This initial matrix vector product defines the initial residual vector 
    r_ = b-t_;

    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    // typename NSL::RT_extractor<Type>::type rsqr_curr = NSL::real( NSL::LinAlg::inner_product(r_,r_) );
    // typename NSL::RT_extractor<Type>::type rsqr_prev = rsqr_curr;
    typename NSL::RT_extractor<Type>::type rsqr_prev = NSL::real( NSL::LinAlg::inner_product(r_,r_) );
    
    // if the guess is already good enough return
    if (rsqr_prev <= errSq_) {
        // NSL::Logger::info("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr),NSL::LinAlg::sqrt(errSq_),0);
        NSL::Logger::info("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_prev),NSL::LinAlg::sqrt(errSq_),0);
        return x_;
    }

    // The initial gradient vector is then given by the residual
    p_ = r_;

    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; ++count){
        // compute the matrix vector product to determine the direction
        // t = M @ p
        t_ = this->Mfree_(p_);

        // determine the scale of the orthogonalization
        //alpha{i} = (r{i},r{i})/(p{i},t{i}) (remember we stored (r{i},r{i}) in rsqr_prev)
        Type alpha = rsqr_prev / NSL::LinAlg::inner_product(p_, t_);

        // update the solution x according to the step
        // x{i+1} = x{i} + alpha{i} * p{i}
        x_ += alpha * p_;

        // compute the residual 
        // r{i+1} = r{i} - alpha{i} * t{i}
        r_ -= alpha * t_;

        // and the resulting error square
        // err = (r{i+1},r{i+1})
        typename NSL::RT_extractor<Type>::type rsqr_curr = NSL::real( NSL::LinAlg::inner_product(r_,r_) );

        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        if (rsqr_curr <= errSq_) {
            NSL::Logger::info("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr),NSL::LinAlg::sqrt(errSq_),count);
            return x_;
        }

        // compute the momentum update scale
        // beta{i} = (r{i+1},r{i+1})/(r{i},r{i}
        typename NSL::RT_extractor<Type>::type beta = rsqr_curr / rsqr_prev;

        // update the momentum
        // p{i+1} = r{i+1} + beta{i} * p{i}
        p_ = r_ + beta * p_;

        // now prepare the previous residual square for the next iteration
        rsqr_prev = rsqr_curr;

        // On debug level we print the solver status every step
        NSL::Logger::debug("CG Iteration: {}/{} | α = {} | ε² = {} |  β = {}", count, maxIter_, NSL::to_string(alpha), rsqr_curr, beta);
    } // for(counter)

    NSL::Logger::error("Error CG did not converge within {} iterations! |r| = {}", maxIter_, NSL::LinAlg::sqrt(rsqr_prev));

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // solve_

template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b ){
    // initialize the solution vector x_ which after convergence 
    // stores the approximate result x = M^{-1} @ b.
    // Multiple initializations are possible and can enhance the convergence
    // see e.g. Preconditioning. Here we just choose a simple start vector
    // which is an arbitrary choise.
    x_ = b;
    return solve_(b);
}

// template<NSL::Concept::isNumber Type >
// NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b , const NSL::Tensor<Type> & x0 ){
//     // initialize the solution vector x_ which after convergence 
//     // stores the approximate result x = M^{-1} @ b.
//     // Multiple initializations are possible and can enhance the convergence
//     // see e.g. Preconditioning. Here we just choose a simple start vector
//     // which is an arbitrary choise.
//     x_ = x0;    
//     return solve_(b);
// }

template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b, const NSL::Tensor<Type> & pre ){
    // initialize the solution vector x_ which after convergence 
    // stores the approximate result x = M^{-1} @ b.
    // Multiple initializations are possible and can enhance the convergence
    // see e.g. Preconditioning. Here we just choose a simple start vector
    // which is an arbitrary choise.
    x_ = b;

    // Compute the initial matrix vector product and store it in the 
    // corresponding vector t
    t_ = this->M_(x_);

    // This initial matrix vector product defines the initial residual vector 
    r_ = b-t_;

    // The initial gradient vector is then given by the residual
    NSL::Tensor<Type> pre_prev = this->solve_(r_);
    pre_prev = this->M_(r_);


    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    typename NSL::RT_extractor<Type>::type rp_curr = NSL::real( NSL::LinAlg::inner_product(r_,p_) );
    typename NSL::RT_extractor<Type>::type rp_prev = rp_curr;
    
    // // if the guess is already good enough return
    // if (rsqr_curr <= errSq_) {
    //     NSL::Logger::info("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr),NSL::LinAlg::sqrt(errSq_),0);
    //     return x_;
    // }

    p_ = pre_prev;

    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; ++count){
        // compute the matrix vector product to determine the direction
        // t = M @ p
        t_ = this->M_(p_);

        // determine the scale of the orthogonalization
        //alpha{i} = (r{i},r{i})/(p{i},t{i}) (remember we stored (r{i},r{i}) in rsqr_prev)
        Type alpha = rp_prev / NSL::LinAlg::inner_product(p_, t_);

        // update the solution x according to the step
        // x{i+1} = x{i} + alpha{i} * p{i}
        x_ += alpha * p_;

        // compute the residual 
        // r{i+1} = r{i} - alpha{i} * t{i}
        r_ -= alpha * t_;

        // and the resulting error square
        // err = (r{i+1},r{i+1})
        NSL::Tensor<Type> pre_curr = this->solve_(r_);

        typename NSL::RT_extractor<Type>::type rp_curr = NSL::real( NSL::LinAlg::inner_product(r_,pre_curr) );
        typename NSL::RT_extractor<Type>::type rsqr_curr = NSL::real( NSL::LinAlg::inner_product(r_,r_) );

        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        if (rsqr_curr <= errSq_) {
            NSL::Logger::info("CG Converged with precision: {} < {} after {} steps", NSL::LinAlg::sqrt(rsqr_curr),NSL::LinAlg::sqrt(errSq_),count);
            return x_;
        }

        // compute the momentum update scale
        // beta{i} = (r{i+1},r{i+1})/(r{i},r{i}
        typename NSL::RT_extractor<Type>::type beta = rp_curr / rp_prev;

        // update the momentum
        // p{i+1} = r{i+1} + beta{i} * p{i}
        p_ = pre_curr + beta * p_;

        // now prepare the previous residual square for the next iteration
        rp_prev = rp_curr;

        // On debug level we print the solver status every step
        NSL::Logger::debug("CG Iteration: {}/{} | α = {} | ε² = {} |  β = {}", count, maxIter_, NSL::to_string(alpha), rsqr_curr, beta);
    } // for(counter)

    NSL::Logger::error("Error CG did not converge within {} iterations! |r| = {}", maxIter_, NSL::LinAlg::sqrt(rp_prev));

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // operator()

} // namespace NSL::LinAlg


#endif //NSL_CG_TPP
