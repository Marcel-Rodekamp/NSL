#ifndef NSL_U1_CORR_TPP
#define NSL_U1_CORR_TPP

namespace NSL::Measure::U1 {

template<NSL::Concept::isNumber Type>
class corr: public Measurement {
    public:
    corr(NSL::Parameter params, NSL::H5IO & h5, 
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
        Nt(params["Nt"].to<NSL::size_t>()),
        gamma_(
            params["device"].to<NSL::Device>(), 
            params["dim"].to<NSL::size_t>()
        )
    {}

    void measure() override;
    void measureBlack();
    void measureRed();

    

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
    NSL::Gamma<Type> gamma_;
}
;

template<NSL::Concept::isNumber Type>
void corr<Type>::measure(){


    NSL::Logger::info("Start Measuring U1::corr");

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
            "/markovChain/{}/corr/",
            cfgID
        );

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has corr, skipping... ", cfgID);
	        continue;
	    } 

        if( cfgID % logFrequency == 0 ){
            NSL::Logger::info(
                "Calculating corr on configuration {}/{}", 
                cfgID, maxCfg
            );
        }

        // read configuration 
        this->h5_.read(U_,fmt::format("{}/markovChain/{}/U",basenode_,cfgID));
        M.populate(U_);

        //correlator is a vecotor along the time axis
        NSL::Tensor<Type> correlator (Nt);
        //setup the cg
        NSL::LinAlg::CG<Type> cg_(M, NSL::FermionMatrix::MMdagger);
        //this is our source
        NSL::Tensor<Type> source (Nt,Nx,2);
        
        //spin down
        source (0,NSL::Slice(0,Nx), 0) = 1;
        NSL::Tensor<Type> MMdaggerInv = cg_(source);
        NSL::Tensor<Type> Minv =  M.Mdagger(MMdaggerInv);
        correlator += (Minv*NSL::LinAlg::conj(Minv)).sum(2).sum(1);

        source = NSL::zeros_like(U_);
        //spin up
        source (0,NSL::Slice(0,Nx), 1) = 1;
        MMdaggerInv = cg_(source);
        Minv =  M.Mdagger(MMdaggerInv);
        correlator += (Minv*NSL::LinAlg::conj(Minv)).sum(2).sum(1);


        NSL::Tensor<Type> c = -correlator/Nx;

        correlator = NSL::zeros_like(correlator);
        Type traceRechts = 0;
        Type secondBlack = 0;

        for (NSL::size_t x = 0; x< Nx; x++){
            source = NSL::zeros_like(U_);
            source(0,x,0) = 1;
            MMdaggerInv = cg_(source);
            Minv =  M.Mdagger(MMdaggerInv);
            traceRechts += NSL::LinAlg::inner_product(source,Minv);
            secondBlack += NSL::LinAlg::inner_product(source,Minv);

            source = NSL::zeros_like(U_);
            source(0,x,1)= 1;
            MMdaggerInv = cg_(source);
            Minv =  M.Mdagger(MMdaggerInv);
            //minus sign because of the gamma five
            traceRechts -= NSL::LinAlg::inner_product(source,Minv);
            secondBlack += NSL::LinAlg::inner_product(source,Minv);

            for (NSL::size_t t = 0; t< Nt; t++){
                source = NSL::zeros_like(U_);
                source(t,x,0) = 1;
                MMdaggerInv = cg_(source);
                Minv =  M.Mdagger(MMdaggerInv);
                correlator(t) += NSL::LinAlg::inner_product(source,Minv);

                source = NSL::zeros_like(U_);
                source(t,x,1)= 1;
                MMdaggerInv = cg_(source);
                Minv =  M.Mdagger(MMdaggerInv);
                //minus sign because of the gamma five
                correlator(t) -= NSL::LinAlg::inner_product(source,Minv);
            }

        }
        std::cout << traceRechts << std::endl;
        std::cout << correlator << std::endl;
        c += correlator * traceRechts /Nx; 

        this->h5_.write(c,basenode_+node);

    } // for cfgID

} // measure()

template<NSL::Concept::isNumber Type>
void corr<Type>::measureBlack(){


    NSL::Logger::info("Start Measuring U1::corrBlack");

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
            "/markovChain/{}/corrBlack/",
            cfgID
        );

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has corrBlack, skipping... ", cfgID);
	        continue;
	    } 

        if( cfgID % logFrequency == 0 ){
            NSL::Logger::info(
                "Calculating corrBlack on configuration {}/{}", 
                cfgID, maxCfg
            );
        }

        // read configuration 
        this->h5_.read(U_,fmt::format("{}/markovChain/{}/U",basenode_,cfgID));
        M.populate(U_);

        //correlator is a vecotor along the time axis
        NSL::Tensor<Type> correlator (Nt);
        //setup the cg
        NSL::LinAlg::CG<Type> cg_(M, NSL::FermionMatrix::MMdagger);
        //this is our source
        NSL::Tensor<Type> source (Nt,Nx,2);
        
        //spin down
        source (0,NSL::Slice(0,Nx), 0) = 1;
        NSL::Tensor<Type> MMdaggerInv = cg_(source);
        NSL::Tensor<Type> Minv =  M.Mdagger(MMdaggerInv);
        correlator += (Minv*NSL::LinAlg::conj(Minv)).sum(2).sum(1);

        source = NSL::zeros_like(U_);
        //spin up
        source (0,NSL::Slice(0,Nx), 1) = 1;
        MMdaggerInv = cg_(source);
        Minv =  M.Mdagger(MMdaggerInv);
        correlator += (Minv*NSL::LinAlg::conj(Minv)).sum(2).sum(1);


        NSL::Tensor<Type> c = -correlator/Nx;

        this->h5_.write(c,basenode_+node);

    } // for cfgID

} // measureBlack()


template<NSL::Concept::isNumber Type>
void corr<Type>::measureRed(){


    NSL::Logger::info("Start Measuring U1::corrBlack");

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
            "/markovChain/{}/corrRed/",
            cfgID
        );

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has corrRed, skipping... ", cfgID);
	        continue;
	    } 

        if( cfgID % logFrequency == 0 ){
            NSL::Logger::info(
                "Calculating corrRed on configuration {}/{}", 
                cfgID, maxCfg
            );
        }

        // read configuration 
        this->h5_.read(U_,fmt::format("{}/markovChain/{}/U",basenode_,cfgID));
        M.populate(U_);

        //correlator is a vecotor along the time axis
        NSL::Tensor<Type> correlator (Nt);
        //setup the cg
        NSL::LinAlg::CG<Type> cg_(M, NSL::FermionMatrix::MMdagger);
        //this is our source
        NSL::Tensor<Type> source (Nt,Nx,2);


        correlator = NSL::zeros_like(correlator);
        Type traceRechts = 0;
        Type secondBlack = 0;

        for (NSL::size_t x = 0; x< Nx; x++){
            source = NSL::zeros_like(U_);
            source(0,x,0) = 1;
            NSL::Tensor<Type> MMdaggerInv = cg_(source);
            NSL::Tensor<Type> Minv =  M.Mdagger(MMdaggerInv);
            traceRechts += NSL::LinAlg::inner_product(source,Minv);
            secondBlack += NSL::LinAlg::inner_product(source,Minv);

            source = NSL::zeros_like(U_);
            source(0,x,1)= 1;
            MMdaggerInv = cg_(source);
            Minv =  M.Mdagger(MMdaggerInv);
            //minus sign because of the gamma five
            traceRechts -= NSL::LinAlg::inner_product(source,Minv);
            secondBlack += NSL::LinAlg::inner_product(source,Minv);

            for (NSL::size_t t = 0; t< Nt; t++){
                source = NSL::zeros_like(U_);
                source(t,x,0) = 1;
                MMdaggerInv = cg_(source);
                Minv =  M.Mdagger(MMdaggerInv);
                correlator(t) += NSL::LinAlg::inner_product(source,Minv);

                source = NSL::zeros_like(U_);
                source(t,x,1)= 1;
                MMdaggerInv = cg_(source);
                Minv =  M.Mdagger(MMdaggerInv);
                //minus sign because of the gamma five
                correlator(t) -= NSL::LinAlg::inner_product(source,Minv);
            }

        }
        std::cout << traceRechts << std::endl;
        std::cout << correlator << std::endl;
        NSL::Tensor<Type> c = correlator * traceRechts /Nx; 

        


        this->h5_.write(c,basenode_+node);

    } // for cfgID

} // 

} // namespace NSL::Measure::U1

#endif // NSL_U1_WILSON_LOOP_TPP
