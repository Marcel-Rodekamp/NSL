#include "NSL.hpp"

template<NSL::Concept::isNumber Type>
bool acceptReject(NSL::Parameter params, const Type & Sprop, const Type & Sprev);

double decreaseStepSize(NSL::Parameter params, double stepSize, double ImSErr);
double increaseStepSize(NSL::Parameter params, double stepSize, double ImSErr);
bool stepSizeCheck(NSL::Parameter params, double & stepSize, double flowTime);

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
    params.addParameter<std::string>(
        "name", yml["system"]["name"].as<std::string>()
    );
    // The inverse temperature 
    params.addParameter<Type>(
        "beta", yml["system"]["beta"].as<double>()
    );
    // The number of time slices
    params.addParameter<NSL::size_t>(
        "Nt", yml["system"]["Nt"].as<NSL::size_t>()
    );
    // The number of ions
    params.addParameter<NSL::size_t>(
        "Nx", yml["system"]["nions"].as<NSL::size_t>()
    );
    // The on-site interaction
    params.addParameter<Type>(
        "U", yml["system"]["U"].as<double>()
    );
    // The chemical potential
    params.addParameter<Type>(
        "mu", yml["system"]["mu"].as<double>()
    );
    // The tangen plane value 
    params.addParameter<double>(
        "tangent plane", yml["system"]["tangent plane"].as<double>()
    );
    // The h5 file name to store the simulation results
    params.addParameter<std::string>(
        "h5file", yml["fileIO"]["h5file"].as<std::string>()
    );
    // The step size for the Runge Kutta
    params.addParameter<double>(
        "step size", yml["RK4"]["step size"].as<double>()
    );
    // minimal step size
    params.addParameter<double>(
        "min step size", yml["RK4"]["min step size"].as<double>()
    );
    // The minimal flow time
    if(yml["RK4"]["min flow time"]){
        params.addParameter<double>(
            "min flow time", yml["RK4"]["min flow time"].as<double>()
        );
    } else {
        // By default we assume at least a single step.
        params.addParameter<double>(
            "min flow time", yml["RK4"]["step size"].as<double>()
        );
    }
    params.addParameter<double>(
        "max flow time", yml["RK4"]["max flow time"].as<double>()
    );
    params.addParameter<double>(
        "adaptive attenuation", yml["RK4"]["adaptive attenuation"].as<double>()
    );
    // The Number of Configurations to generate
    params.addParameter<NSL::size_t>(
        "Nconf", yml["Train Data"]["Nconf"].as<NSL::size_t>()
    );
    // Precision with what the imaginary part must match
    if(yml["RK4"]["ImS precision"]){
        params.addParameter<double>(
            "ImS precision", yml["RK4"]["ImS precision"].as<double>()
        );
    } else {
        params.addParameter<double>(
            "ImS precision", (params["max flow time"].to<double>()/params["step size"].to<double>()) * pow(params["step size"].to<double>(), 3) 
        );
    }

    // Now we want to log the found parameters
    // - key is a std::string name,beta,...
    // - value is a ParameterEntry * which is a wrapper around the actual 
    //   value of interest, we can use ParameterEntry::repr() to get a string
    //   representation of the stored value
    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value->repr() );
    }

    // create an H5 object to store data
    NSL::H5IO h5(params["h5file"].to<std::string>(),NSL::File::Truncate);

    // define the basenode for the h5file, everything is stored in 
    std::string BASENODE;

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    params.addParameter<decltype(lattice)>("lattice", lattice);

    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);

    // define a hubbard fermion action using exponential discretization
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(params);

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration to use during the generation process
    NSL::Configuration<Type> conf{{"phi",
            NSL::Tensor<Type>(
                params["device"].to<NSL::Device>(), 
                params["Nt"].to<NSL::size_t>(), 
                params["Nx"].to<NSL::size_t>()
            )
    }};

    // Define a number of maximal flow steps as a savety stop. 
    NSL::size_t numMaxFlowStep = params["max flow time"].to<double>()/params["step size"].to<double>();

    // Create tensors to store all data we write after all configurations 
    // are computed
    NSL::Tensor<Type> phi_TP(params["device"].to<NSL::Device>(), params["Nconf"].to<NSL::size_t>(), params["Nt"].to<NSL::size_t>(), params["Nx"].to<NSL::size_t>());
    NSL::Tensor<Type> phi_FM(params["device"].to<NSL::Device>(),params["Nconf"].to<NSL::size_t>(),params["Nt"].to<NSL::size_t>(),params["Nx"].to<NSL::size_t>());
    NSL::Tensor<Type> S_TP(params["Nconf"].to<NSL::size_t>());
    NSL::Tensor<Type> S_FM(params["Nconf"].to<NSL::size_t>());
    NSL::Tensor<double> flowTimes(params["Nconf"].to<NSL::size_t>());
    NSL::Tensor<double> mus(params["Nconf"].to<NSL::size_t>());
    NSL::Tensor<double> sigmas(params["Nconf"].to<NSL::size_t>());

    //===================================================================
    //===================================================================
    // This concludes the setup. The next lines define the generation process
    // n=0,1,...,Nconf-1: int; denotes the configuration id 
    // tIDX: int; denotes the time step id
    // flowTime: double; denotes the current flow time associated with the 
    //                   step tIDX.
    //===================================================================
    //===================================================================

    for(NSL::size_t n = 0; n < params["Nconf"].to<NSL::size_t>(); ++n){
        // Under some circumstances a trajectory can not be used as data
        // We then have to draw a new starting configuration and calculate 
        // a new trajectory. In this case redoFlag is set to true and the 
        // n is not increased.
        bool redoFlag = false;

        // define a short hand notation for the step size to avoid writing
        // the "to<double>()" the whole time.
        const double stepSize = params["step size"].to<double>();

        // initialize RK4 step 
        // Set up a single step runge kutta with step size given from the parameters
        // that integrates the holomorphic flow equation:
        // dPhi(t)/dt = ( dS(Phi(t))/dPhi(t) )^*
        NSL::Integrator::RungeKutta4 RKstep(
            /*action=*/S,
            /*maxTime=*/stepSize,
            /*num steps=*/1, 
            /*conjugateGrad=*/true
        );

        // initialize a vector for the actions
        NSL::Tensor<Type> actVals(numMaxFlowStep+1);

        // Sample new configuration from N(mu,sigma) + ic
        // where mu ~ N(0,1)
        //    sigma ~ U( 1/2 deltaU, 1.5 deltaU )
        // on the tangent plane
        mus(n) = NSL::Random::randn<double>();
        sigmas(n) = NSL::Random::rand<double>( 
            0.5 * NSL::real(NSL::Hubbard::tilde<Type>(params,"U")), 
            1.5 * NSL::real(NSL::Hubbard::tilde<Type>(params,"U")) 
        );
        // randomly sampled real part
        conf["phi"].randn(mus(n), sigmas(n));
        // shift to tangent plane
        conf["phi"].imag() = params["tangent plane"].to<double>();

        // store the configuration
        phi_TP(n,NSL::Slice(),NSL::Slice()) = conf["phi"];

        // compute the action of the initial configuration
        actVals(0) = S(conf);

        // define the flow time; each step will add to this variable if accepted
        double flowTime = 0;

        // define another variable used for indexing the flow time step
        // tIDX=0 means no flow
        NSL::size_t tIDX = 1;

        // We want to define an acceptance rate as tIDX / tries where tries
        // is counted at any try to generate a flow step. Further, tries
        // serves as a final run count such that we don't end in a infinite
        // loop it becomes reset when ever we accept a configuration
        NSL::size_t tries = 0; 

        Type actVal = 0;

        // loop until flow time is reached
        for(tIDX=1; tIDX < numMaxFlowStep+1; ++tIDX){ 
        //while(flowTime < params["max flow time"].to<double>()){
            // check that the tries do not run to large
            if(tries >= 1000){
                NSL::Logger::debug("Max tries {} exceeded", tries);
                break;
            }

            // check if flow time is reached
            if(flowTime >= params["max flow time"].to<double>()){break;}

            // check that we do not exceed the max steps required to flow
            // this hit's only if the step size is reduced to much and 
            // stays low
            /*
            if(tIDX >= numMaxFlowStep){ 
                NSL::Logger::debug("Max steps {} exceeded", numMaxFlowStep);

                // only redo the trajectory if we flow to little
                if (flowTime < params["min flow time"].to<double>()){
                    redoFlag=true;
                }
                break;
            } 
            */

            // If the step size is to small we don't continue.
            // If the step is to large RKstep.stepSize() is adjusted
            if(stepSizeCheck(params,RKstep.stepSize(),flowTime)){
                NSL::Logger::debug("Step size < min step size ({} < {})",
                    RKstep.stepSize(), params["min step size"].to<double>()
                );

                break;
            }

            // do a single RK4 step
            NSL::Configuration<Type> proposal = RKstep( conf );

            // compute the action of the new configuration
            actVal = S(proposal);

            // check if we ran into under/overflows. If so we need to redo
            // the trajectory
            if(std::isnan(actVal)){
                NSL::Logger::debug("Found nan action after {} steps, at flow time {}", tIDX, flowTime);

                // pick the earlier time step 
                --tIDX;

                break;
            }

            // By construction the imaginary part of the action remains constant
            // Therefore, we can double check if the integration worked by 
            // compute the action difference modulo 2 pi (represented as exp(i dS ))
            //If Im S is not precise enough, we reduce the step size 
            // by the same rule as isle did it:
            // https://github.com/evanberkowitz/isle/blob/devel/src/isle/cpp/integrator.cpp#109
            double ImSErr = NSL::LinAlg::abs(NSL::LinAlg::exp( i*NSL::imag(actVals(0)-actVal) ) - 1.);

            if(ImSErr > params["ImS precision"].to<double>()){
                NSL::Logger::debug(
                    "ImS precision not reached at configuration {}; ΔImS = {:.2e} > {:.1e}; new step size = {}", 
                    n, ImSErr, params["ImS precision"].to<double>(), RKstep.stepSize() 
                );

                RKstep.stepSize() = decreaseStepSize(params, RKstep.stepSize(), ImSErr);

                // enhance the tries to not run infinitely
                ++tries;

                // precision of integration was to low, redo the step with
                // smaller step size
                continue;
            }

            // we can now check that the step is still relevant for our dataset
            if(acceptReject(params,/*S_prop=*/actVal,/*S_prev=*/actVals(tIDX-1))){
                // accept
                conf = proposal;
                actVals(tIDX) = actVal;

                // add to the flow time
                flowTime += RKstep.stepSize();

                // Log the current status
                NSL::Logger::info("Configuration {}/{}: Accepted flow time {:>.4}(tIDX={}/{}; step size={}) after {} try(s); S = ({:>1.2e}; {:>1.2e})",
                    n+1,params["Nconf"].to<NSL::size_t>(),flowTime,tIDX,numMaxFlowStep,stepSize, tries+1, NSL::real(actVal), NSL::imag(actVal)
                );

                // allow to increase the step size again
                RKstep.stepSize() = increaseStepSize(params, RKstep.stepSize(), ImSErr);

                // reset the tries the tries for the next flow time
                tries = 0;
            } else {
                // Log the current status
                NSL::Logger::info("Configuration {}/{}: Rejected flow time {:>.4}(tIDX={}/{}; step size={}) after {} try(s); S = ({:>1.2e}; {:>1.2e})",
                    n+1,params["Nconf"].to<NSL::size_t>(),flowTime,tIDX,numMaxFlowStep,stepSize, tries+1, NSL::real(actVal), NSL::imag(actVal)
                );

                break;
            }
        } // while flowTime < maxFlowTime and ...

        // if the minimal flow time is not reached redo the trajectory
        if(flowTime < params["min flow time"].to<double>()){redoFlag=true;}
        
        // Redo: We pick a new trajectory starting point for the same 
        // configuration id n.
        if(redoFlag){
            NSL::Logger::debug(
                "Didn't manage to flow. Redoing trajectory {}; flow time: {:>1.4}(tIDX={}, last step size={:>1.2e})", 
                n, flowTime, tIDX, RKstep.stepSize() 
            );
            --n; 
            continue;
        }
    
        // store the results
        phi_FM(n,NSL::Slice(),NSL::Slice()) = conf["phi"];
        flowTimes(n) = flowTime;
        S_FM(n) = actVal; 
        S_TP(n) = actVals(0);
    } // for n = 0,1,...,Nconf-1

    // write the results to file
    h5.write(phi_TP, "phi_TP");
    h5.write(phi_FM, "phi_FM");
    h5.write(S_TP, "S_TP");
    h5.write(S_FM, "S_FM");
    h5.write(flowTimes, "FlowStatistics/flowTimes");
    h5.write(sigmas, "FlowStatistics/sigmas");
    h5.write(mus, "FlowStatistics/locations");

    return EXIT_SUCCESS;
}


template<NSL::Concept::isNumber Type>
bool acceptReject(NSL::Parameter params, const Type & Sprop, const Type & Sprev){
    // Compute the system volume to make the weight 
    double V = params["Nt"].to<NSL::size_t>() * params["Nx"].to<NSL::size_t>();

    //evaluate the weights 
    // w = exp( -Re(S_t - S_{t-1})/V )
    double acceptanceProbability = NSL::LinAlg::exp( 
        -NSL::real(Sprop-Sprev)/V
    );

    // choose to continue or stop flowing,
    // accept with probability U(0,c) where c is the denominator 
    // including earlier flow times
    return NSL::Random::rand<double>(0.,1.) <= acceptanceProbability ;
}

double decreaseStepSize(NSL::Parameter params, double stepSize, double ImSErr){
    stepSize = std::max(  
        stepSize*params["adaptive attenuation"].to<double>() * std::pow(
            params["ImS precision"].to<double>()/ImSErr, 1.0/5.0
        ), // pow
        params["min step size"].to<double>()
    ); // max

    return stepSize;
}

double increaseStepSize(NSL::Parameter params, double stepSize, double ImSErr){
    double newStepSize = stepSize * std::min(
        params["adaptive attenuation"].to<double>() * std::pow(
            params["ImS precision"].to<double>()/ImSErr, 1.0/4.0
        ), //  stepSize' = stepSize * \sqrt{4}{ epsMax / epsFound }
        2.0 // at most double the step size
    );

    if(newStepSize > 0.1){return stepSize;}
    
    return newStepSize;
}

bool stepSizeCheck(NSL::Parameter params, double & stepSize , double flowTime){
    // ensure that the next step does not increase flow time more then 
    // max flow time 
    if( flowTime + stepSize > params["max flow time"].to<double>() ){
        stepSize = params["max flow time"].to<double>() - flowTime;
    } 

    // check that the step size is not to small
    return stepSize < params["min step size"].to<double>();
}

