#include "NSL.hpp"

/*
 * This test runs a leapfrog with multiple Nmd steps (1,11,21,...,91) and 
 * measures the relative energy difference 
 * /f[ 
 *      \left\vert \frac{ H_{old} - H_{new} }{H_{old}} \right\vert
 * /f]
 * Then the data is printed as an array 
 * [
 * [1, $err_1],
 * [11, $err_2],
 * [21, $err_3],
 * ...
 * [91, $err_10],
 * ]
 *
 * which can be copied (by hand!) into a numpy.array. You can simply use
 *
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array(
    $copy in the data from your terminal
)

plt.plot(data[1:,0], (data[0,1]/data[1:,0]), '.:', label = "linear")  # This is put in as a reference
plt.plot(data[1:,0], (data[0,1]/data[1:,0])**2,'.:', label = "square") # This is the one the data should match
plt.plot(data[1:,0], (data[0,1]/data[1:,0])**3, '.:', label = "cubic")  # This is put in as a reference
plt.plot(data[1:,0],data[1:,1]/data[1,1],'.:', label = "data") 
plt.yscale('log')
plt.xscale('log') 
plt.legend()
plt.show()

```
 *
 * */

int main(){
    // Define the parameters of your system
    typedef NSL::complex<double> Type;

    Type I{0,1};

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 0.25;
    //    On-Site Coupling
    Type m    = -1;
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 8;
    //    Dimension of the System
    NSL::size_t dim = 2;
    double trajectoryLength = 1;

    NSL::Parameter params;
    params["beta"]= beta;
    params["bare mass"]=m;
    params["Nt"]=Nt;
    params["Nx"]=Nx;
    params["dim"]=dim;
    params["device"]=NSL::CPU();
    params["Nf"] = 1;

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    // Initialize the action
    NSL::Action::Action S = 
        NSL::Action::PseudoFermionAction<
            Type,decltype(lattice), NSL::FermionMatrix::U1::Wilson<Type>
        >(lattice,params,"U")
        +NSL::Action::U1::WilsonGaugeAction<Type>(params)
    ;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"U", NSL::Tensor<Type>(Nt,Nx,dim)}
    };
    config["U"] = NSL::LinAlg::exp(
        I * NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim)
    );

    S.computePseudoFermion(config);

    NSL::Configuration<Type> momentum{
        {"U", NSL::Tensor<Type>(Nt,Nx,dim)}
    };
    momentum["U"] = NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim);

    // calculate the energy at the starting point of trajectory
    Type H_old = 0.5 * (momentum["U"]*momentum["U"]).sum() + S(config);

    std::cout << "[\n";

    // loop over different molecular dynamics steps
    for (NSL::size_t Nmd = 100; Nmd < 500; Nmd+=25){

        // Initialize the integrator
        NSL::Integrator::U1::Leapfrog leapfrog( 
            /*action*/S,  
            /*trajectoryLength*/trajectoryLength,
            /**numberSteps*/Nmd
        );

        // perform a leapfrog
        auto [pconfig,pmomentum] = leapfrog(config,momentum);

        /*
        std::cout << pconfig["U"].real() << std::endl;
        std::cout << pconfig["U"].imag() << std::endl;
        std::cout << pmomentum["U"].real() << std::endl;
        std::cout << pmomentum["U"].imag() << std::endl;
        */

        // compute the new energy at the end of the trajectory
        Type H_new = 0.5 * (pmomentum["U"]*pmomentum["U"]).sum()
                   + S(pconfig);

        // calculate the error
        NSL::RealTypeOf<Type> err = NSL::LinAlg::abs( (H_old - H_new) / H_old );

        // report the error
        std::cout << "[" << Nmd << ", " << err << "]," << std::endl;
    }

    std::cout << "]\n";

    return EXIT_SUCCESS;
}
