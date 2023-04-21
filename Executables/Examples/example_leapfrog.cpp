#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "NSL.hpp"

int main(){
    
    typedef NSL::complex<double> cd;
    //    NSL::H5IO h5("/Users/tomluu/Research/git/NSL/myTests/data_Nt16.h5");
	
    NSL::size_t Nx =  8;
    NSL::size_t Nt =  32;
    NSL::Tensor<cd> phi(Nt,Nx); phi.randn(); 
    NSL::Tensor<cd> pi(Nt,Nx); pi.randn();
    NSL::Lattice::Ring<cd> lattice(Nx); 

    phi.imag() = 0;
    pi.imag() = 0;


    //  std::vector<std::complex<double>> pphi(Nx*Nt);
    //    std::vector<std::complex<double>> ppi(Nx*Nt);
    //    std::vector<std::complex<double>> H1(20);

    //    HighFive::File h5f("/Users/tomluu/Research/git/NSL/myTests/data_Nt32.h5", HighFive::File::ReadWrite | HighFive::File::OpenOrCreate );
    //    HighFive::DataSet dataset = h5f.getDataSet("phi");
    //    dataset.read(pphi);
    //    dataset = h5f.getDataSet("pi");
    //    dataset.read(ppi);
    //    dataset = h5f.getDataSet("run/H1");
    //    dataset.read(H1);

    /*
    for (int t=0; t<Nt; t++)
      for (int i=0; i<Nx; i++)
	{
	  phi(t,i) = pphi[t*Nx+i];
	  pi(t,i) = ppi[t*Nx+i];
	}       
    */
    
    // define configuration
    NSL::Configuration<cd> config{
		{"phi",phi}, 
    };

    cd beta = 10.0;
    cd U = 3.0;
    NSL::Action::HubbardFermionAction<cd,decltype(lattice),
        NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>>::Parameters paramsHFM(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*lattice=*/lattice
        );
    
    // define momentum
    NSL::Configuration<cd> momentum{
		{"phi",pi}, 
	};

    // define the parameters for the action
    NSL::Action::HubbardGaugeAction<cd>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );

    NSL::Action::HubbardGaugeAction<cd> S_gauge(params);
    NSL::Action::HubbardFermionAction<cd,decltype(lattice),NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>> S_fermion(paramsHFM);
    // define the action
    NSL::Action::Action S = S_gauge + S_fermion;

    cd Hi, Hf;

    Hi = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S(config);

    //    std::cout << std::setprecision(15) << Hi << std::endl;
    // exit(0);

    //    int index = 0;
    for (int Nmd = 10; Nmd < 210; Nmd += 10){
    // define integrator
    NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ 1,
        /*numberSteps=*/ Nmd,
        /*backward*/ false // optional
    );

    // integrate eom
    auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);

    /*
    std::cout << config["phi"] << std::endl;
    std::cout << momentum["phi"] << std::endl;

    std::cout << config_proposal["phi"] << std::endl;
    std::cout << momentum_proposal["phi"] << std::endl;

    std::cout << std::endl;
    */

 
    Hf = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S(config_proposal);
    std::cout << Nmd << "\t" << NSL::LinAlg::abs((Hf-Hi).real()/Hi.real()) << std::endl;
    }

    return EXIT_SUCCESS;
}
