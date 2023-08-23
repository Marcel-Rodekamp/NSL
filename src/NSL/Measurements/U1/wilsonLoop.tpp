#ifndef NSL_U1_WILSON_LOOP_TPP
#define NSL_U1_WILSON_LOOP_TPP

namespace NSL::Measure::U1 {

template<NSL::Concept::isNumber Type>
void planarWilsonLoop(NSL::Tensor<Type> & W, const NSL::Tensor<Type> & P, NSL::size_t Nmu, NSL::size_t mu, NSL::size_t Nnu, NSL::size_t nu){
    for( NSL::size_t mushift = 0; mushift < Nmu; ++mushift ){
        for( NSL::size_t nuShift = 0; nuShift < Nnu; ++nuShift ){
            W *= NSL::LinAlg::shift(NSL::LinAlg::shift(
                    P, nuShift, nu
                ), mushift, mu
            );
        }
    }

    // all data is stored in W
}

template<NSL::Concept::isNumber Type>
class PlanarWilsonLoop: public Measurement {
    public:
    PlanarWilsonLoop(NSL::Parameter params, NSL::H5IO & h5, 
            NSL::size_t nt, 
            NSL::size_t nx, 
            std::string basenode_
    ):
        Measurement(params, h5),
        // toDo generalize for more then 2 dimensions
        U_( 
                params["device"].to<NSL::Device>(), 
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["dim"].to<NSL::size_t>()
        ),
        W_( 
                params["device"].to<NSL::Device>(), 
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
        ),
        P_( 
                params["device"].to<NSL::Device>(), 
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
        ),
        Nmu_(nt),
        Nnu_(nx),
        basenode_(basenode_)
    {}

    //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e. 
    //! Propagators with averaged second time coordinate
    void measure() override;

    protected:
    bool skip_(bool overwrite, std::string node){
        bool exists = this->h5_.exist(fmt::format("{}{}",basenode_,node));

        // if overwrite is specified always calculate the correlator
        if (overwrite){return false;}

        // if correlator doesn't exist always calculate it
        if (not exists){return false;}

        // if correlator exists only recompute if overwrite is true 
        // (this is the only remaining case)
        return true;
    }

    NSL::Tensor<Type> U_;
    NSL::Tensor<Type> W_;
    NSL::Tensor<Type> P_;

    NSL::size_t mu_ = 0; // time like
    NSL::size_t Nmu_;  // number of steps in time direction
    NSL::size_t nu_ = 1; // space like
    NSL::size_t Nnu_; // number of steps in space direction.

    std::string basenode_;
};

template<NSL::Concept::isNumber Type>
void PlanarWilsonLoop<Type>::measure(){
    NSL::Logger::info("Start Measuring U1::WilsonLoop({},{})", Nmu_,Nnu_);

     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(basenode_+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    std::size_t logFrequency = 1;
    if(maxCfg >= 100){
        logFrequency = static_cast<NSL::size_t>( 0.01*maxCfg );
    }

    // Determine the number of time sources:
    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){
        std::string node = fmt::format(
            "/markovChain/{}/WilsonLoop/mu{}Xnu{}/Nmu{}XNnu{}/",
            cfgID, mu_, nu_, Nmu_, Nnu_
        );

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has Wilson Loop, skipping... ", cfgID);
	        continue;
	    } 

        if( cfgID % logFrequency == 0 ){
            NSL::Logger::info(
                "Calculating Planar Wilson Loop (nt={};nx={}); on configuration {}/{}", 
                Nmu_, Nnu_, cfgID, maxCfg
            );
        }

        // read configuration 
        this->h5_.read(U_,fmt::format("{}/markovChain/{}/U",basenode_,cfgID));

        // reset the stored memory
        W_ = 1; 

        P_ = NSL::U1::plaquette(U_,mu_,nu_);

        planarWilsonLoop(W_,P_,Nmu_,mu_,Nnu_,nu_);

        // write the result
        this->h5_.write(W_.mean(),basenode_+node);

    } // for cfgID

} // measure()



} // namespace NSL::Measure::U1

#endif // NSL_U1_WILSON_LOOP_TPP
