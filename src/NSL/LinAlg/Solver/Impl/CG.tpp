#ifndef NSL_CG_TPP
#define NSL_CG_TPP

#include "CG.hpp"

#include "../../../LinAlg/inner_product.tpp"
#include "../../../Tensor/Factory/like.tpp"
#include "complex.hpp"

namespace NSL::LinAlg{
template<NSL::Concept::isNumber Type >
NSL::Tensor<Type> CG<Type>::operator()(const NSL::Tensor<Type> & b ){
    // This algorithm can be found e.g.: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm

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

    // The residual square is given by the square of the residual
    // We require two instances to store the previous (prev) and the current (curr)
    // error (this is a simple efficiency optimization)
    // inner_product returns a number of type `Type` from which the real 
    // part is extracted, the imaginary part is 0 by construction
    typename NSL::RT_extractor<Type>::type rsqr_curr = NSL::real( NSL::LinAlg::inner_product(r_,r_) );
    typename NSL::RT_extractor<Type>::type rsqr_prev = rsqr_curr;
    
    // if the guess is already good enough return
    if (rsqr_curr <= errSq_) {
        return x_;
    }

    // The initial gradient vector is then given by the residual
    p_ = r_;

    // break up condition for maximum number of iteration
    for(NSL::size_t count = 1; count <= maxIter_; ++count){
        // compute the matrix vector product to determine the direction
        // t = M @ p
        t_ = this->M_(p_);

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
        rsqr_curr = NSL::real( NSL::LinAlg::inner_product(r_,r_) );

        // check for convergence agains the errSq_ determined by the 
        // parameter eps (errSq_ = eps*eps) of the constructor to this class
        // if succeeded return the solution x_ = M^{-1} b;
        if (rsqr_curr <= errSq_) {
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

        // comment int for debug purpose
        //! ToDo If compiled in debug mode compile this report for production this report is not interesting.
        //std::cout << "CG Iteration: " << count << "/" << maxIter_
        //          << " | α = " << NSL::to_string(alpha)
        //          << " | ε² = " << rsqr_curr //<< ">" << errSq_  
        //          << " | β = " << beta 
        //          << std::endl;

    } // for(counter)

    // If the CG did not succeed in the given iterations, raise a runtime
    // error which can be caught or terminates the program.
    throw std::runtime_error(
        "Error CG did not converge within "
        +std::to_string(maxIter_)
        +" iterations! |r|^2 = "
        + std::to_string(rsqr_prev)
    );

    // this should never be reached but put it just in case something goes wrong.
    return x_;

} // operator()

} // namespace NSL::LinAlg


#endif //NSL_CG_TPP
