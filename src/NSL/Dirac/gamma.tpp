#ifndef NSL_DIRAC_TPP
#define NSL_DIRAC_TPP

namespace NSL{

template<NSL::Concept::isNumber Type>
class Gamma{
    public:

        Gamma(NSL::Device dev, NSL::size_t dim){
            //! ToDo: Sparse tensors would be nice here!
            NSL::complex<NSL::RealTypeOf<Type>> I{0,1};

            NSL::Tensor<Type> gamma(dim+1,dim,dim);
            if (dim == 2) {
                // pauli_0 = [[0,1],[1,0]]
                gamma(0,0,1) = Type(1);
                gamma(0,1,0) = Type(1);

                // pauli_1 = [[0,-1j],[1j,0]]
                gamma(1,0,1) = Type(-I);
                gamma(1,1,0) = Type(I);

                // pauli_1 = [[1,0],[0,-1]]
                gamma(2,0,0) = Type(1);
                gamma(2,1,1) = Type(-1);
            } else {
                throw std::logic_error(fmt::format("Gamma matrix not implemented for dim = {}",dim));
            }

            gamma_ = gamma.to(dev);
        }
        NSL::Tensor<Type> operator()(NSL::size_t mu){
            return gamma_(mu,NSL::Ellipsis());
        }

    private:

        void initGammaMatrix(NSL::Device dev, NSL::size_t dim){

        }

        NSL::Tensor<Type> gamma_;
};

}

#endif // NSL_DIRAC_TPP
