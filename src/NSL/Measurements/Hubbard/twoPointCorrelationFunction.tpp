#ifndef NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
#define NSL_TWO_POINT_CORRELATION_FUNCTION_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "device.tpp"
#include "parameter.tpp"
#include "../measure.hpp"

namespace NSL::Measure::Hubbard {

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
class TwoPointCorrelator: public Measurement {
    public:
        TwoPointCorrelator(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, NSL::Hubbard::Species species, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            cg_(hfm_, NSL::FermionMatrix::MMdagger),
            species_(species),
	    corrKblock_(
                params["device"].to<NSL::Device>(),
		params["Nt"].to<NSL::size_t>(),
		params["wallSources"].shape(1).to<NSL::size_t>(),
		params["wallSources"].shape(1).to<NSL::size_t>()
            ),
	    corrK_(
                params["device"].to<NSL::Device>(),
                params["wallSources"].shape(1).to<NSL::size_t>(),
		params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            corr_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
	    srcVecK_(
                params["device"].to<NSL::Device>(),
                params["wallSources"].shape(1).to<NSL::size_t>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            srcVec_(
                params["device"].to<NSL::Device>(),
                params["Nx"].to<NSL::size_t>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            phi_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            basenode_(basenode_)
    {}

    TwoPointCorrelator(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5, NSL::Hubbard::Species species):
        TwoPointCorrelator(
            lattice,
            params,
            h5, 
            species,
            params["name"]
        )
    {}

    //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e. 
    //! Propagators with averaged second time coordinate
    void measure() override;

    void measure(NSL::size_t NumberTimeSources);

    void measureK();
    void measureK(NSL::size_t k, NSL::size_t NumberTimeSources);

    protected:
    bool skip_(bool overwrite, std::string node){
        bool exists = this->h5_.exist(fmt::format("{}{}",std::string(basenode_),node));

        // if overwrite is specified always calculate the correlator
        if (overwrite){return false;}

        // if correlator doesn't exist always calculate it
        if (not exists){return false;}

        // if correlator exists only recompute if overwrite is true 
        // (this is the only remaining case)
        return true;
    }

    FermionMatrixType hfm_;
    NSL::LinAlg::CG<Type> cg_;
    NSL::Hubbard::Species species_;

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> corrK_;
    NSL::Tensor<Type> corrKblock_;
    NSL::Tensor<Type> srcVec_;
    NSL::Tensor<Type> srcVecK_;

    NSL::Tensor<Type> phi_;

    std::string basenode_;
};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoPointCorrelator<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources){
    // populate the fermion matrix using the free configuration
    hfm_.populate(phi_,species_);

    // Reset memory
    // - Result correlator
    corr_ = Type(0);
    // - Source vector
    srcVec_ = Type(0);

    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = Nt/NumberTimeSources;

    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
        // Define a point source
        // The Slice here takes out just the single fibre x. We put it 
        // in to return a (device) Tensor from the random access. This 
        // is a hack and should be improved for standard random access.
        srcVec_.index_fill(Type(1), NSL::Range(Nx), tsrc, NSL::Range(Nx));

        // invert MM^dagger
        NSL::Tensor<Type> invMMdag = cg_(srcVec_);

        // back multiply M^dagger to obtain M^{-1}
        // invM is of shape Nx x Nt x Nx
        NSL::Tensor<Type> invM = hfm_.Mdagger(invMMdag);

        // Using a point sink allows to just copy invM as corr(t,y,x)
        // We shift the 1st axis (time-axis) if invM by tsrc and apply anti periodic 
        // boundary conditions
        // shift t -> t - tsrc
        invM.shift( -tsrc, -2, -Type(1) );
        
        // Average over all source times
        corr_ += invM.transpose(0,1).transpose(1,2); 

        // reset source vector
        // Slice: same as above
        //srcVec_(tsrc,NSL::Slice(x,x+1)) = Type(0);
        srcVec_ = Type(0);
    } // tsrc

    corr_ /= Type(NumberTimeSources);
      
} // measure(Ntsrc);

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoPointCorrelator<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t k, NSL::size_t NumberTimeSources){
    // populate the fermion matrix using the free configuration
    hfm_.populate(phi_,species_);

    // Reset memory
    // - Result correlator
    corrK_ = Type(0);
    corrKblock_ = Type(0);
    // - Source vector
    srcVecK_ = Type(0);

    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = Nt/NumberTimeSources;
    
    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
    	// Define a wall source
	    srcVecK_(NSL::Slice(),tsrc,NSL::Slice()) = NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(k,NSL::Slice(),NSL::Slice());

        // invert MM^dagger
        NSL::Tensor<Type> invMMdag = cg_(srcVecK_);

        // back multiply M^dagger to obtain M^{-1}
        // invM is of shape Nx x Nt x Nx
        NSL::Tensor<Type> invM = hfm_.Mdagger(invMMdag);

        // Using a point sink allows to just copy invM as corr(t,y,x)
        // We shift the 1st axis (time-axis) if invM by tsrc and apply anti periodic 
        // boundary conditions
        // shift t -> t - tsrc
        invM.shift( -tsrc, -2, -Type(1) );

        // Average over all source times
        corrK_ += invM; // I changed something here!!!!!

        srcVecK_ = Type(0);
    } // tsrc

    corrK_ /= Type(NumberTimeSources);
    int uDim = params_["wallSources"].shape(1);

    // for (int t = 0;t< Nt;t++){
        for(int sigma1=0; sigma1<uDim; sigma1++){
            NSL::Tensor<Type> wallSource = NSL::Tensor<Type> (params_["wallSources"])(k,sigma1,NSL::Slice()).expand(Nt, 0);
            for(int sigma2=0; sigma2<uDim; sigma2++){
     	        corrKblock_(NSL::Slice(),sigma1,sigma2) = NSL::LinAlg::inner_product(wallSource, corrK_(sigma2,NSL::Slice(),NSL::Slice()), 1);
	        }
        }
//	 corrKblock_(t,0,1) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(k,0,NSL::Slice()),corrK_(1,t,NSL::Slice()));
//	 corrKblock_(t,1,0) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(k,1,NSL::Slice()),corrK_(0,t,NSL::Slice()));
//	 corrKblock_(t,1,1) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(k,1,NSL::Slice()),corrK_(1,t,NSL::Slice()));
    // }

     /*
     for (int t=0;t<Nt;t++) {
     std::cout << "(" << NSL::real(corrKblock_(t,0,0)) << "," << NSL::imag(corrKblock_(t,0,0))<< ")  "
               << "(" << NSL::real(corrKblock_(t,0,1)) << "," << NSL::imag(corrKblock_(t,0,1))<< ")  "
               << "(" << NSL::real(corrKblock_(t,1,0)) << "," << NSL::imag(corrKblock_(t,1,0))<< ")  "
               << "(" << NSL::real(corrKblock_(t,1,1)) << "," << NSL::imag(corrKblock_(t,1,1))<< ")  "
	       << std::endl;
     }
     */
      
} // measureK(k, Ntsrc);


template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoPointCorrelator<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::TwoPointCorrelator");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator 
    std::string node;
    if (species_ == NSL::Hubbard::Particle){
          node = "/NonInteracting/correlators/single/particle";
    } else {
          node = "/NonInteracting/correlators/single/hole";
    }
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
    
        // this stores the result in corr_
        measure(1);

        // write the calculated correlator to file
       h5_.write(corr_,std::string(basenode_)+node);
    } else {
        NSL::Logger::info("Non-interacting correlators already exist");
    }

    // Interacting Correlators
    // Initialize memory for the configurations
     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    bool trimFlag = false;
    // Determine the number of time sources:
    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){
        // this is a shortcut, we don't need to invert if we don't overwrite
        // data

        if (species_ == NSL::Hubbard::Particle){
            node = fmt::format("/markovChain/{}/correlators/single/particle",cfgID);
        } else {
            node = fmt::format("/markovChain/{}/correlators/single/hole",cfgID);
        }

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has correlators, skipping... ", cfgID);
	        trimFlag = true;
            continue;
	    }

        if (trimFlag) {
            this->h5_.trimData(node);
            trimFlag = false;
        }

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // read configuration 
        this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

        // compute the correlator. The result is stored in corr_
        measure(this->params_["Number Time Sources"]);

        // write the result
        this->h5_.write(corr_,std::string(basenode_)+node);
    } // for cfgID

} // measure()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoPointCorrelator<Type,LatticeType,FermionMatrixType>::measureK(){
    NSL::Logger::info("Start Measuring Hubbard::TwoPointCorrelator");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator 
    std::string node;
    if (species_ == NSL::Hubbard::Particle){
          node = "/NonInteracting/correlators/single/particle";
    } else {
          node = "/NonInteracting/correlators/single/hole";
    }
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file

    // measure the non-interacting theory
    // U = 0 <=> phi = 0
    phi_ = Type(0);

    int uDim = params_["wallSources"].shape(0);

    for (int k=0; k< uDim; k++ ){
    	if(!skip_(this->params_["overwrite"], std::string(basenode_)+node+"/k"+std::to_string(k))) {
       	    // this stores the result in corrKblock_
            measureK(k,1);

            // write the calculated correlator to file
       	    h5_.write(corrKblock_,std::string(basenode_)+node+"/k"+std::to_string(k));
	}
    } 


    // Interacting Correlators
    // Initialize memory for the configurations
    // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    // Determine the number of time sources:
    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){
        // this is a shortcut, we don't need to invert if we don't overwrite
        // data

        if (species_ == NSL::Hubbard::Particle){
            node = fmt::format("/markovChain/{}/correlators/single/particle",cfgID);
        } else {
            node = fmt::format("/markovChain/{}/correlators/single/hole",cfgID);
        }

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has correlators, skipping... ", cfgID);
	        continue;
	    } 

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // read configuration 
        this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

	for (int k=0; k< uDim; k++ ){
    	  if(!skip_(this->params_["overwrite"], std::string(basenode_)+node+"/k"+std::to_string(k))) {
       	    // compute the correlator. The result is stored in corrKblock_
            measureK(k,this->params_["Number Time Sources"]);

            // write the calculated correlator to file
       	    this->h5_.write(corrKblock_,std::string(basenode_)+node+"/k"+std::to_string(k));
	  }
        } // for k
    } // for cfgI

} // measureK()

} // namespace NSL::Measure::Hubbard

#endif // NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
