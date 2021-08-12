#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP


namespace NSL::TestingExpDisc {
//ToDo: Is that correct?
template<typename Type>
NSL::TimeTensor<Type> BF( NSL::TimeTensor<Type> & phi, NSL::Tensor<Type> & expKappa) {
    c10::complex<double> num = (0,1);
    NSL::TimeTensor<Type> out = ((NSL::LinAlg::mat_vec(phi,num)).exp().expand(phi.shape(1)))*expKappa;
    int t= phi.shape(0) - 1;
    out[{t}] *= -1;

    /*for (int t = 0; t < phi.shape(0); ++t) {
        if (t == phi.shape(0) - 1){
            out[{phi.shape(0)-1}] *= -1;
        }
    }*/
    return out;
}

template<typename Type>
NSL::TimeTensor<Type> &exp_disc_Mp(
        const NSL::TimeTensor<Type> &phi,
        const NSL::TimeTensor<Type> &psi,
        const NSL::Tensor<Type> &expKappa
) {
    NSL::TimeTensor<Type> out(psi);
    out += NSL::LinAlg::foreach_timeslice(NSL::LinAlg::mat_vec, BF(phi, expKappa), psi.shift(1));
    return out;
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

