#include "NSL.hpp"
#include "logger.hpp"

int main(int argc, char ** argv){
    NSL::Logger::init_logger(argc,argv);

    NSL::Logger::info("Torch Version: {}.{}.{}", 
        TORCH_VERSION_MAJOR, 
        TORCH_VERSION_MINOR, 
        TORCH_VERSION_PATCH 
    );

    NSL::size_t Nx = 12;
    NSL::size_t Nt = 64;
    NSL::complex<double> beta = 10;

    NSL::Logger::info("Initializing Lattice::Ring with {} sites", Nx);
    NSL::Lattice::Ring<NSL::complex<double>> lat(Nx);

    NSL::Logger::info("Initializing Fermion Matrix with Nt={} and beta={}",Nt,beta.real());

    NSL::Tensor<NSL::complex<double>> phi(Nt,Nx); phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
 
    NSL::Logger::info("Running CG to compute MM†.");

    NSL::LinAlg::CG<NSL::complex<double>> cg(M,NSL::FermionMatrix::MMdagger);

    NSL::Tensor<NSL::complex<double>> b(Nt,Nx); b.rand();

    NSL::Tensor<NSL::complex<double>> res = cg(b);

    NSL::Logger::info("Done");
}

