#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "NSL.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <ctime>

int main(int argc, char* argv[]){

    typedef NSL::complex<double> Type;
    torch::InferenceMode guard;

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Example MCMC");
    // an example parameter file is RadialMCMC_example_param.yml
    
    auto init_time = NSL::Logger::start_profile("Initialization");
    
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
    // The number of time slices
    params["Nt"]                = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"]                = yml["system"]["nions"].as<NSL::size_t>();
    // The on-site interaction
    params["U"]                 = yml["system"]["U"].as<double>();
    // The full algorithm's save frequency; i.e. frequency in combined update steps of Nradial radial updates and Nhmc HMC steps 
    params["save frequency"]    = yml["HMC"]["save frequency"].as<NSL::size_t>();
    // The number of Radial Updates per combined step
    if (yml["HMC"]["Nradial"]){
        params["Nradial"]  = yml["HMC"]["Nradial"].as<NSL::size_t>();
    }
    else {
        params["Nradial"] = 0;
    }
    // The number of HMC steps per combined step
    if (yml["HMC"]["Nhmc"]){
        params["Nhmc"]  = yml["HMC"]["Nhmc"].as<NSL::size_t>();
    }
    else {
        params["Nhmc"] = 1;
    }
    // The thermalization length
    params["Ntherm"]            = yml["HMC"]["Ntherm"].as<NSL::size_t>();
    // The number of configurations
    params["Nconf"]             = yml["HMC"]["Nconf"].as<NSL::size_t>();
    // The trajectory length
    params["trajectory length"] = yml["Leapfrog"]["trajectory length"].as<double>();
    // The number of molecular dynamic steps
    params["Nmd"]               = yml["Leapfrog"]["Nmd"].as<NSL::size_t>();
    // The h5 file name to store the simulation results
    params["h5file"]            = yml["fileIO"]["h5file"].as<std::string>();
    // The offset: tangent plane/NLO plane
    if (yml["system"]["offset"]){
        params["offset"]        = yml["system"]["offset"].as<double>();
    } else {
        // DEFAULT: offset = 0
        params["offset"]        = 0.0;
    }
    // Chemical Potential
    if (yml["system"]["mu"]){
        params["mu"]            = yml["system"]["mu"].as<double>();
    } else {
        // DEFAULT: mu = 0
        params["mu"]            = 0.0;
    }

    if (yml["measurements"]["Number Time Sources"]){
        params["Number Time Sources"] = yml["measurements"]["Number Time Sources"].as<NSL::size_t>();
    } else {
        // DEFAULT: Number Time Sources = Nt
        params["Number Time Sources"] = 1;
    }

    // Standard deviation of proposal lognormal distribution in radial udpate
    if (yml["HMC"]["radial scale"]){
        params["radial scale"]    = yml["HMC"]["radial scale"].as<double>();
    } else {
        // DEFAULT: radialScale = 1/Volume
        params["radial scale"]    = 1./(params["Nt"].to<double>()*params["Nx"].to<double>()); 
    }

    if (params["radial scale"].to<double>()==0.0){
        params["Nradial"] = 0;
    }

    // Stability method
    if (yml["stability"] ) {
      params["stability"]    = yml["stability"].as<std::string>();
    }

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);
    
    std::vector<std::vector<std::vector<std::vector<double>>>> kblocks2d;
    std::vector<std::vector<double>> momenta;
   

    // Load momentum blocks if they exist
    if (yml["measurements"]["momenta"]){
      momenta = yml["measurements"]["momenta"].as<std::vector<std::vector<double>>>();
      NSL::Tensor<double> mblocks(momenta.size(),momenta[0].size());
      for (int i=0;i<momenta.size(); i++){
        for (int j=0;j<momenta[0].size();j++){
          mblocks(i,j) = momenta[i][j];
        }
      }
      params["momenta"]=mblocks;
    }

    // Wall sources (blocks x bands x Nx, entries [re,im]) are optional: without them (key
    // missing, null or an empty list) measureK projects onto eigenvectors of the hopping matrix.
    const YAML::Node wsNode = yml["measurements"]["wallSources"];
    if (wsNode && !wsNode.IsNull() && !(wsNode.IsSequence() && wsNode.size() == 0)){
      kblocks2d = wsNode.as<std::vector<std::vector<std::vector<std::vector<double>>>>>();

      // reject ragged/malformed input before indexing into it
      const NSL::size_t nBands = kblocks2d[0].size();
      bool wellFormed = nBands > 0;
      for (const auto & block: kblocks2d) {
        wellFormed = wellFormed && block.size() == nBands;
        for (const auto & vec: block) {
          wellFormed = wellFormed && vec.size() == params["Nx"].to<NSL::size_t>();
          for (const auto & z: vec) { wellFormed = wellFormed && z.size() == 2; }
        }
      }
      if (!wellFormed) {
        NSL::Logger::error("measurements: wallSources must be blocks x bands x Nx({}) x [re,im] with the same number (>0) of bands in every block", params["Nx"].to<NSL::size_t>());
        return EXIT_FAILURE;
      }

      NSL::Tensor<NSL::complex<double>> kblocks(kblocks2d.size(),kblocks2d[0].size(),kblocks2d[0][0].size());
      NSL::Logger::info( "Measuring {} momentum block(s), each with {} band(s) of length {}",kblocks2d.size(),kblocks2d[0].size(),kblocks2d[0][0].size());

      // now populate the momentum blocks with proper complex variables
      for (int i=0; i<kblocks2d.size(); i++){
        for (int j=0; j<kblocks2d[0].size(); j++) {
          for (int k=0; k<kblocks2d[0][0].size(); k++) {
            kblocks(i,j,k) = NSL::complex<double> (kblocks2d[i][j][k][0], kblocks2d[i][j][k][1]);
          }
        }
      }
      params["wallSources"]=kblocks.to(params["device"]);
    } else {
      if (wsNode) {
        NSL::Logger::warn("measurements: wallSources is empty, treating it as absent");
      }
    }

    // Without wall sources measureK projects onto eigenvectors of the hopping matrix K. Their
    // single-particle energies are eps_n = -(lambda_n + mu); measurements: eigenvalueIndices
    // (optional) lists indices into the ascending-eps order (0 = lowest eps, ties by eigh index),
    // otherwise the Fermi window measurements: fermiWindow: {below: 1, above: 0} (defaults) selects
    // the levels HOMO-below .. LUMO+above of the non-interacting filling at mu. A null value or
    // an empty list counts as absent (like wallSources); non-integers, out-of-range or repeated
    // indices are errors.
    const YAML::Node eiNode = yml["measurements"]["eigenvalueIndices"];
    const bool eiEmpty = eiNode && eiNode.IsSequence() && eiNode.size() == 0;
    if (eiEmpty) {
      NSL::Logger::info("measurements: eigenvalueIndices is an empty list, ignored (treated as absent)");
    }
    if (eiNode && !eiNode.IsNull() && !eiEmpty) {
      if (params.contains("wallSources")) {
        NSL::Logger::error("measurements: give either wallSources or eigenvalueIndices, not both");
        return EXIT_FAILURE;
      }
      std::vector<double> idx;
      try {
        idx = eiNode.as<std::vector<double>>();
      } catch (const YAML::Exception & e) {
        NSL::Logger::error("measurements: eigenvalueIndices must be a list of integers ({})", e.what());
        return EXIT_FAILURE;
      }
      const NSL::size_t Nx = params["Nx"].to<NSL::size_t>();
      std::set<double> seen;
      bool valid = true;
      for (double i: idx) { valid = valid && i == std::floor(i) && i >= 0 && i < Nx && seen.insert(i).second; }
      if (!valid) {
        NSL::Logger::error("measurements: eigenvalueIndices must be a list of distinct integers in [0,{})", Nx);
        return EXIT_FAILURE;
      }
      NSL::Tensor<double> idxTensor(idx.size());
      for (NSL::size_t i = 0; i < idx.size(); i++) { idxTensor(i) = idx[i]; }
      params["eigenvalueIndices"] = idxTensor;
      NSL::Logger::info("Projecting onto the hopping eigenvectors with energy indices {} (ascending eps = -(lambda + mu))", fmt::join(idx, ", "));
    }

    // Fermi window (default eigenbasis selection): below, above >= 0 levels around HOMO / LUMO
    const YAML::Node fwNode = yml["measurements"]["fermiWindow"];
    if (fwNode && !fwNode.IsNull()) {
      if (!fwNode.IsMap()) {
        NSL::Logger::error("measurements: fermiWindow must be a map {{below: <int >= 0>, above: <int >= 0>}}");
        return EXIT_FAILURE;
      }
      for (const auto & kv: fwNode) {
        const std::string key = kv.first.as<std::string>();
        if (key != "below" && key != "above") {
          NSL::Logger::error("measurements: fermiWindow has an unknown key '{}' (allowed: below, above)", key);
          return EXIT_FAILURE;
        }
        double v = -1;
        try { v = kv.second.as<double>(); } catch (const YAML::Exception &) { v = -1; }
        if (v < 0 || v != std::floor(v)) {
          NSL::Logger::error("measurements: fermiWindow {} must be a non-negative integer", key);
          return EXIT_FAILURE;
        }
        params[key == "below" ? "fermiWindowBelow" : "fermiWindowAbove"] = (int) v;
      }
      if (params.contains("wallSources") || params.contains("eigenvalueIndices")) {
        NSL::Logger::warn("measurements: fermiWindow is ignored when wallSources or eigenvalueIndices are given");
      }
    }
    if (!params.contains("wallSources") && !params.contains("eigenvalueIndices")) {
      NSL::Logger::info("No wall sources: projecting onto the hopping eigenvectors of the Fermi window HOMO-{} .. LUMO+{} (eps = -(lambda + mu))",
          params.contains("fermiWindowBelow") ? params["fermiWindowBelow"].to<int>() : 1,
          params.contains("fermiWindowAbove") ? params["fermiWindowAbove"].to<int>() : 0);
    }


    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> Sg(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    //
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExpSpinBasis<Type,decltype(lattice)>
      > Sf(lattice,params);
    
    // set stability method if defined in yml file, otherwise default is "QR"
    if (yml["stability"] ) {
      std::string stabilityMethod = params["stability"];
      Sf.hfm_.stabilityMethod = stabilityMethod;// "QR", "DIRECTINVERSE", "SVD"
    }

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = Sg + Sf;

    NSL::size_t Nx =  NSL::size_t(params["Nx"]);
    NSL::size_t Nt =  NSL::size_t(params["Nt"]);


    NSL::Configuration<Type> config{
        {"phi",
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };

    // set seed if desired
    NSL::setSeed(1234);
    
    //! \todo: we really need a proper random interface...
    std::complex<double> I = std::complex<double> (0.0,1.0);

    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = NSL::RealTypeOf<Type>(params["offset"]);

    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );
    
    // initialize 2 point correlation function <p^+_x p_y>
    NSL::Measure::Hubbard::FermionPropagatorSpinBasis<
        Type,
        decltype(lattice),
        NSL::FermionMatrix::HubbardExpSpinBasis<
            Type,decltype(lattice)
        >
    > invM(lattice, params, h5);

    // The propagator measurement is only validated with QR stabilization: its SVD path
    // overflows/produces NaN at large beta and it has no DIRECTINVERSE path (falls back to QR).
    // The action above keeps the yml stability method; the measurement always uses QR.
    if (yml["stability"] ) {
      std::string stabilityMethod = params["stability"];
      if (stabilityMethod != "QR") {
        NSL::Logger::warn("stability: {} is not supported by the spin-basis propagator measurement, using QR instead", stabilityMethod);
      }
    }
    invM.hfm_.stabilityMethod = "QR";

    // Perform calculation of wall-source projected propagator components; measureK throws
    // (before writing) e.g. if the requested projection conflicts with the file's content
    try {
        invM.measureK();
    } catch (const std::exception & e) {
        NSL::Logger::error("propagatorSpinBasis aborted: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
