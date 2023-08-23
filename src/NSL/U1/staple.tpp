#ifndef NSL_U1_STAPLE
#define NSL_U1_STAPLE

//! \file

#include "../Tensor.hpp"
#include "LinAlg.hpp"

namespace NSL::U1{

////! Calculate the Staple given a set of angles
///*!
// * @param phi Angles of the \f$U(1)\f$ variable
// * @param sign Used to compute the staple from plaquette P (+1) or from the 
// *        inverse plaquette \f$P^{-1}\f$ (-1). [Default +1]; 
// *
// * @returns Staple \f$K_\mu(x)\f$ for all x and \f$\mu\f$
// *
// *
// * This function calculates the Staple of a \f$U(1)\f$ theory where every
// * gauge link is defined as 
// * \f[
// *      U_\mu(x) = e^{i \varphi_{\mu}(x)}
// * \f]
// *
// * This function expects the angle \f$\varphi_{\mu}(x)\f$ as a tensor of shape
// * \f$(N_t,N_x,...,N_z, dim)\f$
// * where N_x,...,N_z are the sizes of the dim-1 spatial dimensions and N_t 
// * denotes the size of the temporal direction.
// * dim denotes the number of dimensions (spatial + temporal) of the system
// *
// * The Staple is defined as 
// * \f[\begin{align*}
// *      K_{\mu}(x) &= \sum_{\nu = 0; \nu\neq\mu}^{d-1} \left\{
// *            U_\nu(x+\mu) \cdot U_\mu^{-1}(x+\nu) \cdot U_\nu^{-1}(x)
// *          + U_{\nu}^{-1}(x+\mu-\nu) \cdot U_{\mu}^{-1}(x-\nu) \cdot U_{\nu}(x-\nu)
// *      \right\} \\
// *                 &= \sum_{\nu = 0; \nu\neq\mu}^{d-1} \left\{
// *            \exp\left\{ i\left(\varphi_\nu(x+\mu) - \varphi_\mu(x+\nu) - \varphi_\nu(x) \right)\right\}
// *          + \exp\left\{ i\left(\varphi_{\nu}(x-\nu)-\varphi_{\nu}(x+\mu-\nu) - \varphi_{\mu}(x-\nu)\right)\right\}
// *      \right\}
// * \end{align*}\f]
// * */
//template<NSL::Concept::isNumber Type>
//NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> sumAdjacentStaples(const NSL::Tensor<Type> & phi, int sign = +1){
//
//    NSL::size_t dim = phi.shape(-1);
//
//    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> K = NSL::zeros_like(phi);
//
//    for(NSL::size_t mu = 0; mu < dim; ++mu){
//        for(NSL::size_t nu = mu+1; nu < dim; ++nu){
//            K(NSL::Ellipsis(),mu) += NSL::LinAlg::exp( sign*I*(
//                  NSL::LinAlg::shift( phi(NSL::Ellipsis(), nu),/*shift*/1,/*dim*/mu ) 
//                - NSL::LinAlg::shift( phi(NSL::Ellipsis(), mu),/*shift*/1,/*dim*/nu )
//                -                     phi(NSL::Ellipsis(), nu)
//            )); 
//
//            K(NSL::Ellipsis(),mu) += NSL::LinAlg::exp( sign*I*(
//                  NSL::LinAlg::shift( phi(NSL::Ellipsis(), nu),/*shift*/-1,/*dim*/nu )
//                - NSL::LinAlg::shift( phi(NSL::Ellipsis(), mu),/*shift*/-1,/*dim*/nu )
//                - NSL::LinAlg::shift( 
//                    NSL::LinAlg::shift(phi(NSL::Ellipsis(), nu),/*shift*/1,/*dim*/mu),
//                    /*shift*/-1,/*dim*/nu 
//                  ) 
//            ));
//        } // for nu
//    } // for mu
//
//    return K;
//}

template<NSL::Concept::isNumber Type>
NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> sumAdjacentStaples(const NSL::Tensor<Type> & phi){

    NSL::size_t dim = phi.shape(-1);

    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> K = NSL::zeros_like(phi);

    for(NSL::size_t mu = 0; mu < dim; ++mu){
        for(NSL::size_t nu = 0; nu < dim; ++nu){
            if (mu == nu){continue;}

            K(NSL::Ellipsis(),mu) += 
                  NSL::LinAlg::shift( phi(NSL::Ellipsis(), nu),/*shift*/ 1,/*dim*/mu ) 
                * NSL::LinAlg::shift( phi(NSL::Ellipsis(), mu),/*shift*/ 1,/*dim*/nu ).conj()
                *                     phi(NSL::Ellipsis(), nu).conj()
            ; 

            K(NSL::Ellipsis(),mu) +=
                  NSL::LinAlg::shift( 
                        NSL::LinAlg::shift(
                                         phi(NSL::Ellipsis(),nu),/*shift*/ 1,/*dim*/mu 
                        ),                                       /*shift*/-1,/*dim*/nu
                  ).conj()
                * NSL::LinAlg::shift( phi(NSL::Ellipsis(),mu),/*shift*/-1,/*dim*/nu ).conj()  
                * NSL::LinAlg::shift( phi(NSL::Ellipsis(),nu),/*shift*/-1,/*dim*/nu )  
            ;
        } // for nu
    } // for mu

    //std::cout << "ReK = " << K.real() << std::endl;
    //std::cout << "ImK = " << K.imag() << std::endl;

    return K;
}

}

#endif // NSL_U1_STAPLE
