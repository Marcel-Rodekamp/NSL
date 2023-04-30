#include "NSL.hpp"
#include "logger.hpp"

int main(int argc, char ** argv){
    NSL::Logger::init_logger(argc,argv);

    NSL::Logger::info("Torch Version: {}.{}", 
        TORCH_VERSION_MAJOR, 
        TORCH_VERSION_MINOR 
    );

    auto device = NSL::GPU();
    NSL::Logger::info("Running on: {}", device.repr());

    NSL::size_t Nx = 104;
    NSL::size_t Nt = 128;
    NSL::complex<double> beta = 10;

    NSL::Logger::info("Initializing Lattice::Ring with {} sites", Nx);
    NSL::Lattice::Ring<NSL::complex<double>> lat(Nx);
    lat.to(device);

    NSL::Logger::info("Initializing Fermion Matrix with Nt={} and beta={}",Nt,beta.real());

    NSL::Tensor<NSL::complex<double>> phi(device, Nt,Nx); phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
 
    NSL::Logger::info("Running CG to compute MM†.");

    NSL::LinAlg::CG<NSL::complex<double>> cg(M,NSL::FermionMatrix::MMdagger);

    NSL::Tensor<NSL::complex<double>> b(device,Nt,Nx); b.rand();

    auto cg_time =  NSL::Logger::start_profile("CG");
    NSL::Tensor<NSL::complex<double>> res = cg(b);
    NSL::Logger::stop_profile(cg_time);
    NSL::Logger::info("CG took {}s ",std::get<0>(cg_time));

    NSL::Logger::info("Done");
}

