/*!
 * This example shows how a `NSL::Action` is constructed and used.
 * */
#include <iostream>
#include "NSL.hpp"
#include <cmath>

int main(){
	typedef NSL::complex<double> cd;

	// Define the lattice
    NSL::Lattice::Square<cd> lattice({1,2});
    int NT = 3;
    // Define the parameters
    NSL::Parameter params;
    params["Nt"] = NT;
    params["beta"] = 1.0*NT;
    params["mu"] = 0.0;
    params["U"] = 3.0;

    // Define the fermion matrix
    NSL::FermionMatrix::HubbardExp<cd,NSL::Lattice::Square<cd>> FM(lattice,params);

    // test dMdPhi function of fermion matrix by printing each element
    NSL::Tensor<cd> phi(NT,2);
    // phi[1] = 1.0;
    // phi[2] = 2.0;
    // phi[3] = 3.0;
    FM.populate(phi);

    // for (int u = 0; u < NT; u++){
    //     for (int z = 0; z < 2; z++){
    //         std::cout << std::endl << "derivative of M with respect to phi(" << u << "," << z << "):" << std::endl;
    //         for (int t = 0; t < NT; t++){
    //             for (int x = 0; x < 2; x++){
    //                 NSL::Tensor<cd> left = NSL::zeros_like(phi);
    //                 left(t,x) = 1.0;
    //                 for (int s = 0; s < NT; s++){
    //                     for (int y = 0; y < 2; y++){
    //                         NSL::Tensor<cd> right = NSL::zeros_like(phi);
    //                         right(s,y) = 1.0;
    //                         std::cout << FM.dMdPhi(left,right)(u,z) << "   ";
    //                         std::cout << FM.dMdaggerdPhi(left,right)(u,z) << "   ";
    //                     }
    //                 }
    //                 std::cout << std::endl;
    //             }
    //         }
    //     }
    // }

    NSL::Action::Action S = NSL::Action::HubbardGaugeAction<cd>(params)
                            + NSL::Action::PseudoFermionAction<
                            cd,decltype(lattice), NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>
                            >(lattice, params)
    ;

    NSL::Tensor<cd> phi2(NT,2); phi2.randn(); 
    NSL::Tensor<cd> pi(NT,2); pi.randn();
    phi2.imag() = 0;
    pi.imag() = 0;

    NSL::Configuration<cd> config{
		{"phi",phi2}, 
    };

    NSL::Configuration<cd> momentum{
		{"phi",pi}, 
	};

    cd Hi, Hf;

    S.computePseudoFermion(config);
    
    Hi = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S(config);
    
    for (int Nmd = 1; Nmd < 10001; Nmd *= 2){
      // define integrator
      NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ 2.,
        /*numberSteps=*/ Nmd,
        /*backward*/ false // optional
      );

      // integrate eom
      auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);
      Hf = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S(config_proposal);
      std::cout << Nmd << "\t" << NSL::LinAlg::abs((Hf-Hi).real()/Hi.real()) << std::endl;
    }

    /*
./Executables/Examples/example_Pseudofermion | python3 -c "
import sys
import matplotlib.pyplot as plt

data = [line.split() for line in sys.stdin]
x = [float(point[0]) for point in data]
y = [float(point[1]) for point in data]

plt.loglog(x, y, marker='o')
plt.xlabel('$ N_{md}$')
plt.ylabel('$|\\\\frac{H_f-H_i}{H_i}|$')
plt.title('Trajectory-Length $=2$')
plt.show()"
    */
	return EXIT_SUCCESS;

}
