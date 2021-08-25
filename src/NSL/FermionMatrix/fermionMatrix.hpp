#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIX_HPP


namespace NSL::TestingExpDisc {
//ToDo: Is that correct?
template<typename Type>
NSL::TimeTensor<Type> BF(  NSL::TimeTensor<Type> & phi, NSL::Tensor<Type> & expKappa) {
    NSL::TimeTensor<Type> out(phi.shape());
    out.expand(phi.shape(1));
    c10::complex<double> num = (0,1);
    for(std::size_t t=0; t<phi.shape(0); ++t){
        out[t] =NSL::LinAlg::mat_vec(expKappa,((NSL::LinAlg::mat_vec(phi[t],num)).exp().expand(phi.shape(1))));
    }
    out[phi.shape(0) - 1] = out[phi.shape(0) - 1]* -1;

    return out;
}

template<typename Type>
NSL::TimeTensor<Type> exp_disc_Mp(
         NSL::TimeTensor<Type> &phi,
         NSL::TimeTensor<Type> &psi,
         NSL::Tensor<Type> &expKappa
) {
    NSL::TimeTensor<Type> out1 = NSL::TestingExpDisc::BF(phi, expKappa);
    NSL::TimeTensor<Type> out(phi.shape());
    out.expand(phi.shape(1));
    for(long int t=0; t< phi.shape(0); ++t){
        auto out2 = NSL::LinAlg::shift(psi,1);
        auto out3 = out1[t];
       out[t]=NSL::LinAlg::foreach_timeslice(out3,out2);
    }
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

