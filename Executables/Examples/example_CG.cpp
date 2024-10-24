#include "NSL.hpp"
#include "logger.hpp"

int main(int argc, char ** argv){

    NSL::Parameter params = NSL::init(argc, argv, "Example CG");

    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        if (key == "device" || key == "file" || key == "overwrite") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    NSL::size_t Nx = 4;
    NSL::size_t Nt = 32;
    NSL::complex<double> beta = 1;

    NSL::Logger::info("Initializing Lattice::Ring with {} sites", Nx);
    NSL::Lattice::Ring<NSL::complex<double>> lat(Nx);
    lat.to(params["device"]);

    NSL::Logger::info("Initializing Fermion Matrix with Nt={} and beta={}",Nt,beta.real());

    NSL::Tensor<NSL::complex<double>> phi(params["device"].template to<NSL::Device>(), Nt,Nx); 
    //for(NSL::size_t t = 0; t < Nt; ++t){
    //    for(NSL::size_t x = 0; x < Nx; ++x){
    //        phi(t,x) = (t/(x+1)-x/(t+1)) / Nx;
    //    }
    //}
    phi.randn();
    phi *= NSL::LinAlg::sqrt((beta/Nt)*0.1); 
    NSL::FermionMatrix::HubbardExp M(lat,Nt,beta);
    M.populate(phi,NSL::Hubbard::Species::Particle);

    //phi.randn();
    NSL::FermionMatrix::HubbardExp MnonInt(lat,Nt,beta);
    MnonInt.populate(NSL::zeros_like(phi),NSL::Hubbard::Species::Particle);
 
    NSL::Logger::info("Running CG to compute MM†@x = b.");

    //NSL::LinAlg::CGpreconditioned<NSL::complex<double>> cg(
    //    [&M](const NSL::Tensor<NSL::complex<double>> & psi      ) { return M.MMdagger(psi); },
    //    [&MnonInt](const NSL::Tensor<NSL::complex<double>> & psi) { return MnonInt.MMdagger(psi); },
    //    1e-12,
    //    20000
    //);
    NSL::LinAlg::CG<NSL::complex<double>> cg(
        [&M](const NSL::Tensor<NSL::complex<double>> & psi      ) { return M.MMdagger(psi); },
        1e-12,
        20000
    );

    NSL::Tensor<NSL::complex<double>> b(params["device"].template to<NSL::Device>(),Nt,Nx); //b.rand();
    b(0,NSL::Slice(0,Nx)) = 1;

    auto cg_time =  NSL::Logger::start_profile("CG");
    NSL::Tensor<NSL::complex<double>> res = cg(b);
    NSL::Logger::stop_profile(cg_time);
    NSL::Logger::info("CG took {}s ",std::get<0>(cg_time));

    NSL::Logger::info("Done");
}

