#include "NSL.hpp"



int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;
    NSL::complex<double> I{0,1};

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Schwinger Model WF logDet");
    // an example parameter file is can be found in example_param.yml
    
    // Now all parameters are stored in yml, we want to translate them 
    // into the parameter object
    // We can read in the parameter file and put the read data into the 
    // params object, notice this uses the example_param.yml file
    // For personal files, this code needs to be adjusted accordingly
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // convert the data from example_param.yml and put it into the params
    // The name of the physical system
    params["name"]              = yml["system"]["name"].as<std::string>();
    // The inverse temperature 
    params["beta"]              = yml["system"]["beta"].as<double>();
    // bare mass 
    params["bare mass"]         = yml["system"]["bare mass"].as<double>();
    // The number of time slices
    params["Nt"] = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"] = yml["system"]["Nx"].as<NSL::size_t>();
    // dimension of the system 
    params["dim"] = 2;
     
    params["Nf"] = 1;
    


    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });

    
    NSL::FermionMatrix::U1::Wilson<Type> M (
        lattice, params
    );



 
    NSL::Tensor<Type> U = NSL::LinAlg::exp(
        I*NSL::Tensor<double>(
            NSL::size_t(params["Nt"]), NSL::size_t(params["Nx"]), NSL::size_t(params["dim"])
        ).rand(0,2*3.1415)
    );
    NSL::size_t Nx = params["Nx"].to<NSL::size_t>();
    NSL::size_t Nt = params["Nt"].to<NSL::size_t>();
    std::cout << "Real U0:" << std::endl;


      std::cout << "[" << std::endl;
      for (int t = 0; t < params["Nt"]; ++t) {
        std::cout << "[";
        for (int x = 0; x < params["Nx"]; ++x) {
            if (x == (Nx-1)){
                std::cout << U(t,x,0).real();
            }else{
                std::cout << U(t,x,0).real() << ",";
            }
            
        }
        if (t == (Nx-1)){
            std::cout << "]" << std::endl;
        }else{
            std::cout << "]," << std::endl;
        }
        
    }
    std::cout << "]" << std::endl;

    std::cout << "Imag U0:" << std::endl;


      std::cout << "[" << std::endl;
      for (int t = 0; t < params["Nt"]; ++t) {
        std::cout << "[";
        for (int x = 0; x < params["Nx"]; ++x) {
            if (x == (Nx-1)){
                std::cout << U(t,x,0).imag();
            }else{
                std::cout << U(t,x,0).imag() << ",";
            }
            
        }
        if (t == (Nx-1)){
            std::cout << "]" << std::endl;
        }else{
            std::cout << "]," << std::endl;
        }
        
    }
    std::cout << "]" << std::endl;

    std::cout << "Real U1:" << std::endl;


      std::cout << "[" << std::endl;
      for (int t = 0; t < params["Nt"]; ++t) {
        std::cout << "[";
        for (int x = 0; x < params["Nx"]; ++x) {
            if (x == (Nx-1)){
                std::cout << U(t,x,1).real();
            }else{
                std::cout << U(t,x,1).real() << ",";
            }
            
        }
        if (t == (Nx-1)){
            std::cout << "]" << std::endl;
        }else{
            std::cout << "]," << std::endl;
        }
        
    }
    std::cout << "]" << std::endl;

        std::cout << "Imag U1:" << std::endl;


      std::cout << "[" << std::endl;
      for (int t = 0; t < params["Nt"]; ++t) {
        std::cout << "[";
        for (int x = 0; x < params["Nx"]; ++x) {
            if (x == (Nx-1)){
                std::cout << U(t,x,1).imag();
            }else{
                std::cout << U(t,x,1).imag() << ",";
            }
            
        }
        if (t == (Nx-1)){
            std::cout << "]" << std::endl;
        }else{
            std::cout << "]," << std::endl;
        }
        
    }
    std::cout << "]" << std::endl;


    
    M.populate(U);

    std::cout << M.gradLogDetM()(3,3,0).real() << std::endl;

    
    


    // //chi = NSL::randn_like(config.at(this->configKey_), 0., 0.7071067811865476 );
    // NSL::Tensor<Type> chi = NSL::randn_like(U,0., (0.7071067811865476));


    // std::cout << M.M(chi).reshape(NSL::size_t(params["Nt"])*NSL::size_t(params["Nx"])*2).real() << std::endl;
    // std::cout << M.M(chi).reshape(NSL::size_t(params["Nt"])*NSL::size_t(params["Nx"])*2).imag() << std::endl;
    // std::cout << "_______________" << std::endl;

    // std::cout << chi.reshape(NSL::size_t(params["Nt"])*NSL::size_t(params["Nx"])*2).real() << std::endl;
    // std::cout << chi.reshape(NSL::size_t(params["Nt"])*NSL::size_t(params["Nx"])*2).imag() << std::endl;

    // std::cout << "_______________" << std::endl;



    return EXIT_SUCCESS;
}


