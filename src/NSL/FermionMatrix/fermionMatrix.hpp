#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP


namespace NSL::TestingExpDisc {
//ToDo: Is that correct?
template<typename Type>
NSL::TimeTensor<Type> BF(  NSL::TimeTensor<Type> & phi, NSL::Tensor<Type> & expKappa) {
    const long Nt = phi.shape(0);
    const long Nx = phi.shape(1);

    NSL::TimeTensor<Type> out({Nt, Nx, Nx});

    c10::complex<double> num = (0,1);
    for(std::size_t t=0; t<phi.shape(0); ++t){
        out[t]=(((phi[t]*num).exp().expand(phi.shape(1))).prod(expKappa));
    }
    out[0] *= -1;

    return out;
}

template<typename Type>
NSL::TimeTensor<Type> exp_disc_Mp(
         NSL::TimeTensor<Type> &phi,
         NSL::TimeTensor<Type> &psi,
         NSL::Tensor<Type> &expKappa
) {
    NSL::TimeTensor<Type> M= NSL::TestingExpDisc::BF(phi, expKappa);
    NSL::TimeTensor<Type> out(psi.shape());

    for(std::size_t t = 0; t < psi.shape(0); ++t){
        out[t] = NSL::LinAlg::mat_vec(M[t],(NSL::LinAlg::shift(psi,1))[t]);
    }

    return psi - out;
}

} // namespace NSL::TestingExpDisc

// 1. NSL::LinAlg::mat_vec
// 2. NSL::LinAlg::expand
// 3. class Tensor{...};
//    3.1 Copy constructor
// 4. class TimeTensor{...};
//    4.2 NSL::Tensor & shift(offset)
// 5. foreach_timeslice:
// functor on each time slice
/*
 */
#endif //NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP*//*

