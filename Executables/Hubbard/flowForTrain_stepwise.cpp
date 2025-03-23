#include "NSL.hpp"

double decreaseStepSize(NSL::Parameter params, double stepSize, double ImSErr);
double increaseStepSize(NSL::Parameter params, double stepSize, double ImSErr);

NSL::complex<double> i{0,1};

/*! Usage  */
int main(int argc, char ** argv){
    typedef NSL::complex<double> Type;
    
    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Flow Random Configurations For Generating Training Data");
    
    auto init_time = NSL::Logger::start_profile("Initialization");
    
    // Now all parameters are stored in yml, we want to translate them 
    // into the parameter object
    // We can read in the parameter file and put the read data into the 
    // params object, notice this uses the example_param.yml file
    // For personal files, this code needs to be adjusted accordingly
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // convert the data from example_param.yml and put it into the params
    // The name of the physical system
    params["name"] = yml["system"]["name"].as<std::string>();
    // The inverse temperature 
    params["beta"] = yml["system"]["beta"].as<double>();
    // The number of time slices
    params["Nt"] = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"] = yml["system"]["nions"].as<NSL::size_t>();
    // The on-site interaction
    params["U"] = yml["system"]["U"].as<double>();
    // The chemical potential
    params["mu"] = yml["system"]["mu"].as<double>();
    // The tangent plane value 
    params["tangent plane"] = yml["system"]["offset"].as<double>();
    // The h5 file name to store the simulation results
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();
    // The step size for the Runge Kutta
    params["step size"] = yml["RK4"]["step size"].as<double>();
    // minimal step size
    params["min step size"] = yml["RK4"]["min step size"].as<double>();
    // The minimal flow time
    if(yml["RK4"]["min flow time"]){
        params["min flow time"] = yml["RK4"]["min flow time"].as<double>();
    } else {
        // By default we assume at least a single step.
        params["min flow time"] = yml["RK4"]["step size"].as<double>();
    }
    params["max flow time"] = yml["RK4"]["max flow time"].as<double>();
    params["adaptive attenuation"] = yml["RK4"]["adaptive attenuation"].as<double>();
    // The Number of Configurations to generate
    params["Nconf"] = yml["Train Data"]["Nconf"].as<NSL::size_t>();
    params["npoints"] = yml["RK4"]["npoints"].as<NSL::size_t>();
    // Precision with which the imaginary part must match
    if(yml["RK4"]["ImS precision"]){
        params["ImS precision"] = yml["RK4"]["ImS precision"].as<double>();
    } else {
        params["ImS precision"] = (params["max flow time"].to<double>()/params["step size"].to<double>()) * pow(params["step size"].to<double>(), 3);
    }

    // Now we want to log the found parameters
    for(auto [key, value]: params){
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    // Create an H5 object to store data
    NSL::H5IO h5(params["h5file"].to<std::string>(), NSL::File::Truncate);

    // Initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // Define actions
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(lattice, params);
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration to use during the generation process
    NSL::Configuration<Type> conf{{"phi",
            NSL::Tensor<Type>(
                params["device"].template to<NSL::Device>(), 
                params["Nt"].template to<NSL::size_t>(), 
                params["Nx"].template to<NSL::size_t>()
            )
    }};

    // Define npoints (total number of points including initial configuration)
    NSL::size_t npoints = params["npoints"].template to<NSL::size_t>();

    // Create tensors to store all data we write after all configurations are computed
    NSL::Tensor<Type> phi_FM(params["device"].template to<NSL::Device>(), params["Nconf"].template to<NSL::size_t>(), npoints, params["Nt"].template to<NSL::size_t>(), params["Nx"].template to<NSL::size_t>());
    NSL::Tensor<Type> S_FM(params["Nconf"].template to<NSL::size_t>(), npoints);
    NSL::Tensor<double> mus(params["Nconf"].template to<NSL::size_t>());
    NSL::Tensor<double> sigmas(params["Nconf"].template to<NSL::size_t>());

    // Main loop over configurations
    for(NSL::size_t n = 0; n < params["Nconf"].to<NSL::size_t>(); ++n){
        bool redoFlag = false;

        // Sample initial configuration
        mus(n) = NSL::Random::randn<double>();
        sigmas(n) = NSL::Random::rand<double>( 
            0.5 * NSL::real(NSL::Hubbard::tilde<Type>(params,"U")), 
            1.5 * NSL::real(NSL::Hubbard::tilde<Type>(params,"U")) 
        );
        conf["phi"].randn(mus(n), sigmas(n));
        conf["phi"].real() = 1.*(n-params["Nconf"].template to<NSL::size_t>()/2)/params["Nconf"].template to<NSL::size_t>();
        conf["phi"].imag() = params["tangent plane"].template to<double>();

        // Store initial configuration in phi_FM at index 0
        phi_FM(n, 0, NSL::Slice(), NSL::Slice()) = conf["phi"];

        // Compute action of initial configuration
        Type actVal = S(conf);
        S_FM(n, 0) = actVal;

        // Initialize flow variables
        double totalFlowTime = params["max flow time"].to<double>();
        double delta_t = totalFlowTime / (npoints - 1); // Subtract 1 because initial point is already stored

        double flowTime = 0.0;

        // Initial step size
        double stepSize = params["step size"].template to<double>();

        // For each save point after the initial configuration
        for(NSL::size_t point_idx = 1; point_idx < npoints; ++point_idx){
            // Flow from flowTime to flowTime + delta_t
            double targetTime = flowTime + delta_t;

            // Initialize RK4 integrator
            NSL::Integrator::RungeKutta4 RKstep(
                /*action=*/S,
                /*maxTime=*/delta_t,
                /*num steps=*/1, 
                /*conjugateGrad=*/true
            );

            // Set the initial step size
            RKstep.stepSize() = stepSize;

            // Perform integration from flowTime to targetTime
            while(flowTime < targetTime){
                // Remaining time
                double remainingTime = targetTime - flowTime;

                // Adjust step size if necessary
                if(RKstep.stepSize() > remainingTime){
                    RKstep.stepSize() = remainingTime;
                }

                // Check if step size is too small
                if(RKstep.stepSize() < params["min step size"].template to<double>()){
                    NSL::Logger::debug("Step size too small at configuration {}, point {}", n, point_idx);
                    redoFlag = true;
                    break;
                }

                // Perform RK4 step
                NSL::Configuration<Type> proposal = RKstep( conf );

                // Compute action
                Type newActVal = S(proposal);

                // Check for NaNs
                if(std::isnan(NSL::real(newActVal)) or std::isnan(NSL::imag(newActVal))){
                    NSL::Logger::debug("Found NaN action at configuration {}, point {}", n, point_idx);
                    redoFlag = true;
                    break;
                }

                // Check ImS error
                double ImSErr = NSL::LinAlg::abs(NSL::LinAlg::exp( i*NSL::imag(newActVal - actVal) ) - 1.);

                if(ImSErr > params["ImS precision"].template to<double>()){
                    NSL::Logger::debug(
                        "ImS precision not reached at configuration {}, point {}; ΔImS = {:.2e} > {:.1e}; decreasing step size", 
                        n, point_idx, ImSErr, params["ImS precision"].template to<double>()
                    );

                    RKstep.stepSize() = decreaseStepSize(params, RKstep.stepSize(), ImSErr);
                    continue; // Retry with smaller step size
                } else {
                    // Increase step size if possible
                    RKstep.stepSize() = increaseStepSize(params, RKstep.stepSize(), ImSErr);
                }

                // Accept step
                conf = proposal;
                actVal = newActVal;
                flowTime += RKstep.stepSize();
            }

            // If redoFlag is set, break
            if(redoFlag){
                // If at least the first flow point is reached, proceed
                if(point_idx == 1){
                    NSL::Logger::debug("Couldn't reach first flow point for configuration {}; redoing", n);
                    --n; // redo this configuration
                }
                break;
            }

            // Save configuration at this point
            phi_FM(n, point_idx, NSL::Slice(), NSL::Slice()) = conf["phi"];
            S_FM(n, point_idx) = actVal;

            // Log progress
            NSL::Logger::info("Configuration {}/{}: Reached point {}/{}; flowTime = {:.4f}; stepSize = {:.2e}", 
                n+1, params["Nconf"].template to<NSL::size_t>(), point_idx, npoints - 1, flowTime, RKstep.stepSize());

            // Update step size for next interval
            stepSize = RKstep.stepSize();
        }

    } // end for n

    // Write the results to file
    h5.write(phi_FM, "phi_FM"); // Contains initial and flowed configurations
    h5.write(S_FM, "S_FM");     // Contains action values at initial and flowed configurations
    h5.write(sigmas, "FlowStatistics/sigmas");
    h5.write(mus, "FlowStatistics/locations");

    return EXIT_SUCCESS;
}

double decreaseStepSize(NSL::Parameter params, double stepSize, double ImSErr){
    stepSize = stepSize * params["adaptive attenuation"].template to<double>() * std::pow(
            params["ImS precision"].template to<double>() / ImSErr, 1.0 / 5.0
        );
    return stepSize;
}

double increaseStepSize(NSL::Parameter params, double stepSize, double ImSErr){
    double newStepSize = stepSize * params["adaptive attenuation"].template to<double>() * std::pow(
            params["ImS precision"].template to<double>() / ImSErr, 1.0 / 4.0
        );
    if(newStepSize > 0.1){ return stepSize; } // Limit the maximum step size increase
    return newStepSize;
}
