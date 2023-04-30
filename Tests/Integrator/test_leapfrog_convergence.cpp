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
    typedef double Type;

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 1;
    //    On-Site Coupling
    Type U    = 1;
    //    Number of time slices
    NSL::size_t Nt = 2;
    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    // Leapfrog Parameters
    //      Trajectory Length
    NSL::RealTypeOf<Type> trajectoryLength = 1.; // We ensure that this is a real number in case Type is complex
    
    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    //std::cout   << "Setting up a Hubbard-Gauge action with beta=" << beta
    //           << ", Nt=" << Nt << ", U=" << U << " on a ring with " << Nx << " sites." << std::endl;
    // Put the action parameters into the appropriate container
    NSL::Action::HubbardGaugeAction<Type>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );

    // Initialize the action
    NSL::Action::Action S{
        NSL::Action::HubbardGaugeAction<Type>(params)
    };

    // Initialize a configuration as starting point for the trajectory
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].randn();

    // Initialize a momentum as starting point for the trajectory
    NSL::Configuration<Type> momentum{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    momentum["phi"].randn();

    // calculate the energy at the starting point of trajectory
    NSL::RealTypeOf<Type> H_old = 0.5 * (momentum["phi"]*momentum["phi"]).sum() + S(config);

    std::cout << "[\n";

    // loop over different molecular dynamics steps
    for (NSL::size_t Nmd = 1; Nmd < 100; Nmd+=10){
        // Initialize the integrator
        NSL::Integrator::Leapfrog leapfrog( 
            /*action*/S,  
            /*trajectoryLength*/trajectoryLength,
            /**numberSteps*/Nmd
        );

        // perform a leapfrog
        auto [pconfig,pmomentum] = leapfrog(config,momentum);

        // compute the new energy at the end of the trajectory
        NSL::RealTypeOf<Type> H_new = 0.5 * (pmomentum["phi"]*pmomentum["phi"]).sum()
                                    + S(pconfig);

        // calculate the error
        NSL::RealTypeOf<Type> err = NSL::LinAlg::abs( (H_old - H_new) / H_old );

        // report the error
        std::cout << "[" << Nmd << ", " << err << "]," << std::endl;
    }

    std::cout << "]\n";

    return EXIT_SUCCESS;
}
