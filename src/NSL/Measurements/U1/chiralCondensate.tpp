#ifndef NSL_U1_CHIRAL_CONDENSATE_TPP
#define NSL_U1_CHIRAL_CONDENSATE_TPP

namespace NSL::Measure::U1 {

template<NSL::Concept::isNumber Type>
class chiralCondensate: public Measurement {
    public:
    chiralCondensate(NSL::Parameter params, NSL::H5IO & h5, 
            std::string basenode_
    ):Measurement(params, h5),
    U_( 
                params["device"].to<NSL::Device>(), 
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["dim"].to<NSL::size_t>()
        ),
        basenode_(basenode_),
        Nx(params["Nx"].to<NSL::size_t>()),
        Nt(params["Nt"].to<NSL::size_t>())
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
    std::string basenode_;
    NSL::size_t Nx;
    NSL::size_t Nt;
    NSL::Tensor<Type> U_;
}
;

template<NSL::Concept::isNumber Type>
void chiralCondensate<Type>::measure(){
    NSL::Logger::info("Start Measuring U1::chiralCondensate");

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
        NSL::Lattice::Square<Type> lattice({
        Nt,
        Nx
    });

        NSL::FermionMatrix::U1::Wilson<Type> M (
        lattice, params_
    );

    



    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){
        std::string node = fmt::format(
            "/markovChain/{}/chiralCon/",
            cfgID
        );

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has CC, skipping... ", cfgID);
	        continue;
	    } 

        if( cfgID % logFrequency == 0 ){
            NSL::Logger::info(
                "Calculating CC on configuration {}/{}", 
                cfgID, maxCfg
            );
        }

        // read configuration 
        this->h5_.read(U_,fmt::format("{}/markovChain/{}/U",basenode_,cfgID));
        M.populate(U_);
        Type trace = 0;
        for(NSL::size_t i = 0; i < Nt*Nx*2; i++){
            NSL::Tensor<Type> v (Nt,Nx,2);
            v[i] = 1;
            NSL::LinAlg::CG<Type> cg_(M, NSL::FermionMatrix::MMdagger);

        // compute MMdagger * pseudoFermion
         NSL::Tensor<Type> MMdaggerInv = cg_(v);
         NSL::Tensor<Type> Minv =  M.Mdagger(MMdaggerInv);
         trace += NSL::LinAlg::inner_product(v,Minv);
        }
        Type CC= trace/Nx/Nt;


        // write the result
        this->h5_.write(CC,basenode_+node);

    } // for cfgID

} // measure()



} // namespace NSL::Measure::U1

#endif // NSL_U1_WILSON_LOOP_TPP
