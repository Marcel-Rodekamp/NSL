#ifndef NSL_U1_PLAQUETTE
#define NSL_U1_PLAQUETTE

//! \file

#include "../Tensor.hpp"
#include "LinAlg.hpp"

namespace NSL::U1{


//! Calculate the plaquette given a set of angles
/*!
 * @param phi Angles of the \f$U(1)\f$ variable
 * @param mu first direction
 * @param nu second direction
 *
 * @returns Angle of Plaquette \f$ -i \log(P_{\mu\nu}(x)) \f$
 *
 *
 * This function calculates the plaquette of a \f$U(1)\f$ theory where every
 * gauge link is defined as 
 * \f[
 *      U_\mu(x) = e^{i \varphi_{\mu}(x)}
 * \f]
 *
 * This function expects the angle \f$\varphi_{\mu}(x)\f$ as a tensor of shape
 * \f$(N_t,N_x,...,N_z, dim)\f$
 * where N_x,...,N_z are the sizes of the dim-1 spatial dimensions and N_t 
 * denotes the size of the temporal direction.
 * dim denotes the number of dimensions (spatial + temporal) of the system
 *
 * The Plaquette is defined as 
 * \f[\begin{align*}
 *      P_{\mu\nu}(x) &= U_\mu(x) \cdot U_{\nu}(x+\mu) 
 *                     \cdot U_{\mu}^{-1}(x+\nu) \cdot U_{\nu}^{-1}(x)  \\ 
 *                    &= \exp\left\{i \left( \varphi_\mu(x) + \varphi_{\nu}(x+\mu) 
 *                    - \varphi_{\mu}(x+\nu) - \varphi_{\nu}(x) \right)\right\}
 * \end{align*}\f]
 * */
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> plaquette(const NSL::Tensor<Type> & phi, NSL::size_t mu, NSL::size_t nu){
    return phi(NSL::Ellipsis(), mu)
         * NSL::LinAlg::shift( phi(NSL::Ellipsis(), nu),/*shift*/1,/*dim*/mu )
         * 1./NSL::LinAlg::shift( phi(NSL::Ellipsis(), mu),/*shift*/1,/*dim*/nu )
         * 1./phi(NSL::Ellipsis(), nu);
}

}

#endif // NSL_U1_PLAQUETTE
