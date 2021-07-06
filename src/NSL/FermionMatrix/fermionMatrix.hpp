#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP

namespace NSL {
namespace TestingExpDisc {

/* Not ready yet, I comented it out so that you can compile your code
template<typename Type>
NSL::TimeTensor<Type> &BF(NSL::TimeTensor<Type> phi, NSL::Tensor<Type> expKappa) {
    // ToDo: construct out
    NSL::TimeTensor<Type> out(phi.Nt(); ...);

    for (std::size_t t = 0; t < phi.Nt(); ++t) {
        out[t] = expKappa * NSL::Tensor::expand(NSL::LinAlg::exp(1i * phi[t]));

        if (t == phi.Nt() - 1) {
            out[t] *= -1;
        }
    }

    return out;
}

template<typename Type>
NSL::TimeTensor<Type> &exp_disc_Mp(
        const NSL::TimeTensor<Type> &phi,
        const NSL::TimeTensor<Type> &psi,
        const NSL::Tensor<Type> &expKappa,
) {

    const std::size_t Nt = phi.shape(0);
    const std::size_t Nx = phi.shape(1);

    NSL::TimeTensor<Type> out(psi);

    out += NSL::LinAlg::foreach_timeslice(NSL::LinAlg::mat_vec, BF(phi, expKappa), psi.shift(1));

    return out;
}

} // namespace TestingExpDisc
} // namespace NSL
*/


// 1. NSL::LinAlg::mat_vec
// 2. NSL::LinAlg::expand
// 3. class Tensor{...};
//    3.1 Copy constructor
// 4. class TimeTensor{...};
//    4.2 NSL::Tensor & shift(offset)
// 5. foreach_timeslice:
/* // functor on each time slice
 * template<typename Type>
 * auto NSL::LinAlg::foreach_timeslice(
 *      std::function<ReturnType(NSL::TimeTensor<Type>&, NSL::TimeTensor<Type>&)> functor,
 *      NSL::TimeTensor & left,
 *      NSL::TimeTensor & right)
 * {
 *      NSL::TimeTensor out(left.Nt(),...);
 *      for(std::size_t t = 0; t < left.Nt(); ++t){
 *          out[t] = functor(left[t],right[t]);
 *      }
 * }
 * */

#endif //NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP
