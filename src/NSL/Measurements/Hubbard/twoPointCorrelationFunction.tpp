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
        TwoPointCorrelator(NSL::Parameter params, NSL::H5IO & h5, NSL::Hubbard::Species species, std::string basenode_):
            Measurement(params, h5),
            hfm_(params),
            cg_(hfm_, NSL::FermionMatrix::MMdagger),
            species_(species),
            corr_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            srcVec_(
                params["device"].to<NSL::Device>(),
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

    TwoPointCorrelator(NSL::Parameter params, NSL::H5IO & h5, NSL::Hubbard::Species species):
        TwoPointCorrelator(
            params, 
            h5, 
            species,
            fmt::format("{}",params["name"].repr())
        )
    {}

    //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e. 
    //! Propagators with averaged second time coordinate
    void measure() override;

    void measure(NSL::size_t NumberTimeSources);

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

    FermionMatrixType hfm_;
    NSL::LinAlg::CG<Type> cg_;
    NSL::Hubbard::Species species_;

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> srcVec_;

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
    hfm_.populate(phi_);

    // Reset memory
    // - Result correlator
    corr_ = Type(0);
    // - Source vector
    srcVec_ = Type(0);
    
    NSL::size_t tsrcStep = this->params_["Nt"].to<NSL::size_t>()/NumberTimeSources;

    for(NSL::size_t tsrc = 0; tsrc<this->params_["Nt"].to<NSL::size_t>(); tsrc+=tsrcStep){
        for(NSL::size_t x = 0; x < this->params_["Nx"].to<NSL::size_t>(); ++x){
            // Define a point source
            // The Slice here takes out just the single fibre x. We put it 
            // in to return a (device) Tensor from the random access. This 
            // is a hack and should be improved for standard random access.
            srcVec_(tsrc,NSL::Slice(x,x+1)) = Type(1);            

            // invert MM^dagger
            NSL::Tensor<Type> invMMdag = cg_(srcVec_);

            // back multiply M^dagger to obtain M^{-1}
            NSL::Tensor<Type> invM = hfm_.Mdagger(invMMdag);

            // Using a point sink allows to just copy invM as corr(t,y,x)
            // We shift the 0th axis (time-axis) if invM by tsrc and apply anti periodic 
            // boundary conditions

            // shift t -> t - tsrc
            invM.shift( -tsrc );
            // apply anti periodic boundary
            invM(NSL::Slice(this->params_["Nt"].to<NSL::size_t>()-tsrc)) *= -1;

            // Average over all source times
            corr_(NSL::Slice(),NSL::Slice(),x) += invM; 

            // reset source vector
            // Slice: same as above
            srcVec_(tsrc,NSL::Slice(x,x+1)) = Type(0);
        } // x
    } // tsrc

    corr_ /= Type(NumberTimeSources);
      
} // measure(Ntsrc);


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
       h5_.write(corr_,basenode_+node);
    } else {
        NSL::Logger::info("Non-interacting correlators already exist");
    }

    // Interacting Correlators
    // Initialize memory for the configurations

     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(basenode_+"/markovChain");
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
        this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",basenode_,cfgID));

        // compute the correlator. The result is stored in corr_
        measure(this->params_["Number Time Sources"]);

        // write the result
        this->h5_.write(corr_,basenode_+node);
    } // for cfgID

} // measure()

} // namespace NSL::Measure::Hubbard

#endif // NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
