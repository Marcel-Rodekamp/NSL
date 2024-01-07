#include <ATen/core/ATen_fwd.h>
#include <chrono>
#include "NSL.hpp"
#include "complex.hpp"
#include "highfive/H5File.hpp"
#include <ios>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <yaml-cpp/yaml.h>

#include "transformers.tpp"

c10::complex<double> im{0,1};


template<typename Type, typename ActionType>
struct ActionImpl : torch::autograd::Function<ActionImpl<Type,ActionType>> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor phi,
        ActionType * S
    ) {
        if (phi.dim() == 2) {
            phi = phi.unsqueeze(0);
        }

        NSL::size_t Nconf = phi.size(0);

        ctx->save_for_backward({phi});
        savedS[ctx] = S;

        auto phi_NSL = NSL::Tensor<Type>(phi);

        NSL::Tensor<Type> actVals( Nconf );

        for (NSL::size_t n = 0; n < Nconf; ++n){
            NSL::Configuration<Type> conf{{
                "phi",phi_NSL(n,NSL::Slice(),NSL::Slice())
            }};
            actVals(n) = S->eval(conf);
        }

        return torch::Tensor(actVals).requires_grad_(phi.requires_grad());
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list gradOutput
    ) {
        auto phi_ =ctx->get_saved_variables()[0];

        std::cout << "gradOutput = " << gradOutput << std::endl;

        NSL::Tensor<Type> gradInput( phi_.size(0), phi_.size(1), phi_.size(2) );

        for (NSL::size_t n = 0; n < phi_.size(0); ++n){
            NSL::Configuration<Type> conf{{
                "phi",NSL::Tensor<Type>(phi_.index({n,NSL::Slice(),NSL::Slice()}))
            }};

            gradInput(n,NSL::Slice(),NSL::Slice()) = NSL::LinAlg::conj(savedS[ctx]->grad(conf)["phi"]);
        }

        return {torch::Tensor(gradInput) * gradOutput[0]};
    }

    static std::unordered_map<torch::autograd::AutogradContext * , ActionType *> savedS;
};

template<typename Type, typename ActionType>
std::unordered_map<torch::autograd::AutogradContext *, ActionType *> ActionImpl<Type,ActionType>::savedS;

template<typename Type, typename ActionType>
struct Action : torch::nn::Module {
    
    Action(ActionType & S): S(S) {}
    
    torch::Tensor forward(torch::Tensor phi){
//        return ActionImpl<Type,ActionType>::apply(phi, &S);
        if (phi.dim() == 2) {
            phi = phi.unsqueeze(0);
        }

        NSL::size_t Nconf = phi.size(0);

        auto phi_NSL = NSL::Tensor<Type>(phi);
            
        torch::Tensor actVals = torch::zeros( {Nconf}, phi.options() );

        for (NSL::size_t n = 0; n < Nconf; ++n){
            NSL::Configuration<Type> conf{{
                "phi",phi_NSL(n,NSL::Slice(),NSL::Slice())
            }};
            actVals.index_put_({n}, S.eval(conf));
        }

        return torch::Tensor(actVals);
        //return torch::Tensor(actVals).requires_grad_(phi.requires_grad());
    }

    ActionType S;
};

template<typename Type, typename ActionType>
struct StatPower: torch::nn::Module {

    StatPower(ActionType & S_) : S(S_) {}

    torch::Tensor forward(torch::Tensor phi, torch::Tensor origActVals, torch::Tensor logDetJ){
        // compute the new action values
        torch::Tensor newActVals = S.forward(phi);

        // compute the phase
        torch::Tensor phase = torch::exp(
            -torch::real( newActVals-origActVals ) + logDetJ - im*torch::imag( newActVals )
        );

        // return the absolute value of the average phase (i.e the statistical power)
        return torch::abs( torch::mean(phase) );
    }
    
    Action<Type,ActionType> S;
};

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Example MCMC");
    // an example parameter file is can be found in example_param.yml
    
    auto init_time = NSL::Logger::start_profile("Initialization");

    

    // ================================================================================================================
    // ================================================================================================================
    // Set up parameter
    // ================================================================================================================
    // ================================================================================================================


    
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
    params.addParameter<Type>(
        "mu", yml["system"]["mu"].as<double>()
    );
    // Number of training samples 
    params.addParameter<NSL::size_t>(
        "Nconf", yml["HMC"]["Nconf"].as<NSL::size_t>()
    );
    // The training steps length
    params.addParameter<NSL::size_t>(
        "Nepoch", yml["Train"]["Nepoch"].as<NSL::size_t>()
    );
    // The learning rate
    params.addParameter<double>(
        "learning rate", yml["Train"]["learning rate"].as<double>()
    );
    params.addParameter<std::string>(
        "h5file", yml["fileIO"]["h5file"].as<std::string>()
    );
    params.addParameter<double>(
        "offset", yml["system"]["offset"].as<double>()
    );
    
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



    // ================================================================================================================
    // ================================================================================================================
    // Set up physics
    // ================================================================================================================
    // ================================================================================================================



    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);
    params.addParameter<decltype(lattice)>("lattice", lattice);


    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a {}.", 
        NSL::real( Type(params["beta"]) ),
        params["Nt"].to<NSL::size_t>(),
        NSL::real( Type(params["U"])),
        std::string( params["name"] )
    );

    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(params);

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = S_gauge + S_fermion;



    // ================================================================================================================
    // ================================================================================================================
    // Read in training data
    // ================================================================================================================
    // ================================================================================================================



    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string basenode_train(fmt::format("{}/Training",params["name"].repr()));
    std::string basenode_hmc(fmt::format("{}",params["name"].repr()));

    NSL::Tensor<Type> trainConfs_(
        params["Nconf"].to<NSL::size_t>(), params["Nt"].to<NSL::size_t>(), params["Nx"].to<NSL::size_t>()
    );
    NSL::Tensor<Type> trainActVals_(
        params["Nconf"].to<NSL::size_t>()
    );

    for(NSL::size_t n = 0; n < params["Nconf"].to<NSL::size_t>(); ++n){
        NSL::Tensor<Type> tmp(params["Nt"].to<NSL::size_t>(), params["Nx"].to<NSL::size_t>());
        h5.read(
            tmp,
            fmt::format( "{}/markovChain/{}/phi", basenode_hmc, n )
        );

        trainConfs_(n,NSL::Slice(),NSL::Slice()) = tmp;

        Type actVal = 0;
        h5.read(actVal, fmt::format( "{}/markovChain/{}/actVal", basenode_hmc, n ));
        trainActVals_(n) = actVal;
    }
    
    trainConfs_.to(params["device"].to<NSL::Device>());
    trainActVals_.to(params["device"].to<NSL::Device>());

    // convert to torch tensor
    torch::Tensor trainConfs = trainConfs_;
    torch::Tensor trainActVals = trainActVals_;

    // ================================================================================================================
    // ================================================================================================================
    // Set up the training process
    // ================================================================================================================
    // ================================================================================================================



    // Set up of the Neural Network 
    Shift<Type> NN( 
        params["offset"], 
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>() 
    );
    
    // Set up of the training loss 
    StatPower<Type,decltype(S)> loss(S);

    // set up of the optimizer
     torch::optim::Adam optimizer(
        NN.parameters(), 
        /*lr=*/params["learning rate"].to<double>()
    );


    // ================================================================================================================
    // ================================================================================================================
    // Training Process
    // ================================================================================================================
    // ================================================================================================================


    
    torch::Tensor statPower_train = torch::zeros({params["Nepoch"].to<NSL::size_t>()});
    torch::Tensor statPower_valid = torch::zeros({params["Nepoch"].to<NSL::size_t>()});
    for(NSL::size_t epoch = 0; epoch < params["Nepoch"].to<NSL::size_t>(); ++epoch){
        optimizer.zero_grad();

        // do a training step
        auto [pred,logDetJ] = NN.forward(trainConfs);
        auto statPower = loss.forward(pred, trainActVals, logDetJ);
        statPower.backward();
        optimizer.step();

        statPower_train[epoch] = statPower.item();
        NSL::Logger::info(
            "Training Epoch: {} \t StatPower: {}", epoch, statPower.item<double>()
        );
        //std::cout << NN.shift << std::endl;
        std::cout << "NN.shift.grad = " << NN.shift.grad() << std::endl;

        // evaluate the network
    }


    return EXIT_SUCCESS;
}




// ====================================================================================================================
// ====================================================================================================================
// Function Implementations
// ====================================================================================================================
// ====================================================================================================================

