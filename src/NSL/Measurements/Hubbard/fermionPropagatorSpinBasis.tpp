#ifndef NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
#define NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP

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
class FermionPropagatorSpinBasis: public Measurement {
    public:
        FermionPropagatorSpinBasis(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            corr_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
	    corrK_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(), // bands
                params["wallSources"].shape(1).template to<NSL::size_t>() // bands
            ),
            phi_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            basenode_(basenode_),
	invAp1F_U_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	invAp1F_D_(lattice.device(), 1, lattice.sites()),
	invAp1F_T_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	expKdiag_(lattice.device(), lattice.sites()),
	Uk_(lattice.device(), lattice.sites(), lattice.sites()),
	Vk_(lattice.device(), lattice.sites(), lattice.sites()),
	PI_U_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    PI_D_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),
    PI_V_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_U_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_D_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),
    SIGMA_V_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    Fk_U_(lattice.device(),lattice.sites(),lattice.sites()),
    Fk_D_(lattice.device(),lattice.sites()),   
	Fk_V_(lattice.device(),lattice.sites(),lattice.sites()),
	uu_(lattice.device(), lattice.sites(), lattice.sites()),
	dd_(lattice.device(), lattice.sites()),
	vv_(lattice.device(), lattice.sites(), lattice.sites()),
	vu_(lattice.device(), lattice.sites(), lattice.sites()),
	Qnew_(lattice.device(), lattice.sites(), lattice.sites()),
	Dnew_(lattice.device(), lattice.sites()),
	Vnew_(lattice.device(), lattice.sites(), lattice.sites()),
    Qnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	Dnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites()),
	Vnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    vut_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    invAp1F_Ut_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	invAp1F_Dt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites()),
	invAp1F_Tt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    Fk_Ut_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    Fk_Dt_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),   
	Fk_Vt_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
	uut_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	ddt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites()),
	vvt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    Qnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites(), lattice.sites()),
	Dnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites()),
	Vnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites(), lattice.sites())
    {}

    FermionPropagatorSpinBasis(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
        FermionPropagatorSpinBasis(
            lattice,
            params,
            h5, 
            params["name"]
        )
    {}

    //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e.
    //! Propagators with averaged second time coordinate
    void measure() override;
    void measure(NSL::size_t NumberTimeSources, NSL::size_t config);
    void measureK();
    void measureK(NSL::size_t NumberTimeSources, NSL::size_t config);
    
    void measureDiagonal();
    void measureRow(NSL::size_t tsrc);
    void measureColumn(NSL::size_t tsrc);

    void calcPiSigma(NSL::size_t tsrc);

    // maybe make these private later. . .
    NSL::Tensor<Type> invAp1F_U_;
    NSL::Tensor<Type> invAp1F_D_;
    NSL::Tensor<Type> invAp1F_T_;
    NSL::Tensor<Type> expKdiag_, Uk_, Vk_;
    NSL::Tensor<Type> PI_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> PI_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> PI_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> SIGMA_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> SIGMA_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> SIGMA_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> Fk_U_; //(device,Nx,Nx);
    NSL::Tensor<Type> Fk_D_; //(device,Nx);   
    NSL::Tensor<Type> Fk_V_; //(device,Nx,Nx);
    NSL::Tensor<Type> uu_; //(device, Nx, Nx);
    NSL::Tensor<Type> dd_; //(device, Nx);
    NSL::Tensor<Type> vv_; //(device, Nx, Nx);
    NSL::Tensor<Type> vu_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnew_; //(device, Nx);
    NSL::Tensor<Type> Vnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnewt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnewt_; //(device, Nx);
    NSL::Tensor<Type> Vnewt_; //(device, Nx, Nx);
    NSL::Tensor<Type> vut_; //(device, Nx, Nx);
    NSL::Tensor<Type> invAp1F_Ut_;
    NSL::Tensor<Type> invAp1F_Dt_;
    NSL::Tensor<Type> invAp1F_Tt_;
    NSL::Tensor<Type> Fk_Ut_; //(device,Nx,Nx);
    NSL::Tensor<Type> Fk_Dt_; //(device,Nx);   
    NSL::Tensor<Type> Fk_Vt_; //(device,Nx,Nx);
    NSL::Tensor<Type> uut_; //(device, Nx, Nx);
    NSL::Tensor<Type> ddt_; //(device, Nx);
    NSL::Tensor<Type> vvt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnewtt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnewtt_; //(device, Nx);
    NSL::Tensor<Type> Vnewtt_; //(device, Nx, Nx);

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
    NSL::Hubbard::Spin spin_;

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> phi_;

    NSL::Tensor<Type> corrK_;

    NSL::Tensor<Type> wallSink_;
    NSL::Tensor<Type> wallSrc_;

    std::string basenode_;
    std::string leaf;

};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(){
    NSL::Logger::info("Start Measuring Momentum Hubbard::FermionPropagatorSpinBasis");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    std::string node;

    // write the momenta out
    //if (!h5_.exist(std::string(basenode_)+"/Momenta")){
    //    NSL::Tensor<double> momenta = params_["momenta"];
    //    this->h5_.write(momenta,std::string(basenode_)+"/Momenta");
    //}

    // write the non interacting correlator
    node = "/NonInteracting/propagator";

// this is a shortcut, we don't need to calculate the non-interacting
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
        measureK(1,0);
    }else{
        NSL::Logger::info("Non-interacting two-body propagator already exists");
    }

    // Interacting Correlators
    // Initialize memory for the configurations
     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];

    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // read configuration
	this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

        // compute the correlator for the differen time sources
        measureK(this->params_["Number Time Sources"], cfgID);
    } // for cfgID

} // measureK()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    int kDim = params_["wallSources"].shape(0);  // !! got rid of factor of 2
    int bDim = params_["wallSources"].shape(1);

    // Reset memory
    // - Result correlator
    corr_ = Type(0);
    corrK_ = Type(0);

    std::string node;

    bool PiSigma;

    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);

    NSL::Tensor<Type> wallSources(params_["device"].template to<NSL::Device>(), NSL::Tensor<Type> (params_["wallSources"]).shape(0), NSL::Tensor<Type> (params_["wallSources"]).shape(1), NSL::Tensor<Type> (params_["wallSources"]).shape(2));
    wallSources(NSL::Ellipsis()) = NSL::Tensor<Type> (params_["wallSources"]);
//    wallSources(NSL::Slice(0,NSL::none_t(), 2), NSL::Ellipsis()) = NSL::Tensor<Type> (params_["wallSources"]);
//    wallSources(NSL::Slice(1,NSL::none_t(), 2), NSL::Ellipsis()) = NSL::LinAlg::conj(NSL::Tensor<Type> (params_["wallSources"]));

    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
        PiSigma = true;

        // first do up species
        spin_ = NSL::Hubbard::Up;

        // calculate column
    if ((phi_ == Type(0)).all()) { // if non-interacting
        node = fmt::format("/NonInteracting/measurements/spinUp/Ck[t,{}]",tsrc); // node for column
    }else{
        node = fmt::format("/markovChain/{}/measurements/spinUp/Ck[t,{}]",cfgID,tsrc); // node for column
    }
        if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
           if (PiSigma) {
              // populate the fermion matrix using the free configuration
              hfm_.populate(phi_,spin_);
              calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
              PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
           }
       measureColumn(tsrc);
       for (int kSrc=0; kSrc<kDim; kSrc++ ) {
                for (int sigmaSink=0; sigmaSink<bDim; sigmaSink++) {
                    for (int sigmaSrc=0; sigmaSrc<bDim; sigmaSrc++) {
                        corrK_(NSL::Slice(),sigmaSink,sigmaSrc) = NSL::LinAlg::inner_product( wallSources(kSrc,sigmaSink,NSL::Slice()) , NSL::LinAlg::mat_mul(corr_, wallSources(kSrc,sigmaSrc,NSL::Slice())),1);
                    }
                }
                leaf = fmt::format("/{}-{}",kSrc,kSrc); // node for column
                this->h5_.write(corrK_,std::string(basenode_)+node+leaf);  // write out the column
        } // for kSrc

        }else{
        NSL::Logger::info("Config #{} already has up columns, skipping... ", cfgID);
    }

    // here we just measure the charge since it is easy to do
        if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
        if ((phi_ == Type(0)).all()) { // if non-interacting
            node = fmt::format("/NonInteracting/measurements/spinUp/Q"); // node for diagonal
        }else{
            node = fmt::format("/markovChain/{}/measurements/spinUp/Q",cfgID); // node for diagonal
        }
           if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
              if (PiSigma) {
                 // populate the fermion matrix using the free configuration
                 hfm_.populate(phi_,spin_);
                 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
                 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
              }
              measureDiagonal();

	      auto Q = phi_[0]*0.0;
	      for (int t=0; t< Nt; t++) {
		 Q += NSL::LinAlg::diag(corr_(t,NSL::Ellipsis())).sum();
	      }
	      Q /= Nt;
	      this->h5_.write(Q,std::string(basenode_)+node);  // write out the charge

          }else{
        NSL::Logger::info("Config #{} already has up diagonal, skipping... ", cfgID);
    }
        }

        PiSigma = true; // reset bool for the downs. . .

        // now repeat for down species
        spin_ = NSL::Hubbard::Down;

        // populate the fermion matrix using the free configuration
    hfm_.populate(phi_,spin_);

        // calculate column
    if ((phi_ == Type(0)).all()) { // if non-interacting
        node = fmt::format("/NonInteracting/measurements/spinDown/Ck[t,{}]",tsrc); // node for column
    }else{
        node = fmt::format("/markovChain/{}/measurements/spinDown/Ck[t,{}]",cfgID,tsrc); // node for column
    }
        if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
           if (PiSigma) {
              // populate the fermion matrix using the free configuration
              hfm_.populate(phi_,spin_);
              calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
              PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
           }
       measureColumn(tsrc);
       for (int kSrc=0; kSrc<kDim; kSrc++ ) {
                for (int sigmaSink=0; sigmaSink<bDim; sigmaSink++) {
                    for (int sigmaSrc=0; sigmaSrc<bDim; sigmaSrc++) {
                        corrK_(NSL::Slice(),sigmaSink,sigmaSrc) = NSL::LinAlg::inner_product( wallSources(kSrc,sigmaSink,NSL::Slice()) , NSL::LinAlg::mat_mul(corr_, wallSources(kSrc,sigmaSrc,NSL::Slice())),1);
                    }
                }
                leaf = fmt::format("/{}-{}",kSrc,kSrc); // node for column
                this->h5_.write(corrK_,std::string(basenode_)+node+leaf);  // write out the column
        } // for kSrc

        }else{
        NSL::Logger::info("Config #{} already has down columns, skipping... ", cfgID);
    }

	// and now the charge again. . .
        if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
        if ((phi_ == Type(0)).all()) { // if non-interacting
            node = fmt::format("/NonInteracting/measurements/spinDown/Q"); // node for diagonal
        }else{
            node = fmt::format("/markovChain/{}/measurements/spinDown/Q",cfgID); // node for column
        }
           if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
              if (PiSigma) {
                 // populate the fermion matrix using the free configuration
                 hfm_.populate(phi_,spin_);
                 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
                 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
              }
              measureDiagonal();
	      auto Q = phi_[0]*0.0;
              for (int t=0; t< Nt; t++) {
                 Q += NSL::LinAlg::diag(corr_(t,NSL::Ellipsis())).sum();
              }
              Q /= Nt;
              this->h5_.write(Q,std::string(basenode_)+node);  // write out the charge

          }else{
        NSL::Logger::info("Config #{} already has down diagonal, skipping... ", cfgID);
    }
      }

    } // tsrc

} // measureK(Ntsrc,config);	  

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID){

    // Reset memory
    // - Result correlator
    corr_ = Type(0);

    std::string node;

    bool PiSigma;

    // read configuration 
    this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);

    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
        PiSigma = true;

        // first do up spin
	spin_ = NSL::Hubbard::Up;

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/spinUp/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}
	
	// calculate row
	node = fmt::format("/markovChain/{}/propagator/spinUp/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/spinUp/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,spin_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	  }
	}

	PiSigma = true; // reset bool for the downs. . .
	
	// now repeat for down spin
	spin_ = NSL::Hubbard::Down;
		
	// populate the fermion matrix using the free configuration
    	hfm_.populate(phi_,spin_);

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/spinDown/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}

	// calculate row
	node = fmt::format("/markovChain/{}/propagator/spinDown/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/spinDown/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,spin_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	   }
      }

    } // tsrc
      
} // measure(Ntsrc,config);

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::FermionPropagatorSpinBasis");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator
    std::string nodeD,nodeC,nodeR;
    spin_ = NSL::Hubbard::Up;
    nodeD = "/NonInteracting/propagator/spinUp/invM[t,t]";
    nodeC = "/NonInteracting/propagator/spinUp/invM[t,0]";
    nodeR = "/NonInteracting/propagator/spinUp/invM[0,t]";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],nodeD)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
	hfm_.populate(phi_,spin_);
	calcPiSigma(1); // calculuate prefix/suffix terms with tsrc
	measureDiagonal();
	h5_.write(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
	measureRow(1);
	h5_.write(corr_,std::string(basenode_)+nodeR);  // write out the row
	measureColumn(1);
	h5_.write(corr_,std::string(basenode_)+nodeC);  // write out the column
     
    } else {
        NSL::Logger::info("Non-interacting invM[t,t] already exists");
    }

    spin_ = NSL::Hubbard::Down;
    nodeD = "/NonInteracting/propagator/spinDown/invM[t,t]";
    nodeC = "/NonInteracting/propagator/spinDown/invM[t,0]";
    nodeR = "/NonInteracting/propagator/spinDown/invM[0,t]";
    // this is a shortcut, we don't need to calculate the non-interacting
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],nodeD)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
        hfm_.populate(phi_,spin_);
        calcPiSigma(1); // calculuate prefix/suffix terms with tsrc
        measureDiagonal();
        h5_.write(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
        measureRow(1);
        h5_.write(corr_,std::string(basenode_)+nodeR);  // write out the row
        measureColumn(1);
        h5_.write(corr_,std::string(basenode_)+nodeC);  // write out the column

    } else {
        NSL::Logger::info("Non-interacting invM[t,t] already exists");
    }


    // Interacting Correlators
    // Initialize memory for the configurations
     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // compute the correlator for the differen time sources
        measure(this->params_["Number Time Sources"], cfgID);

    } // for cfgID

} // measure()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureDiagonal(){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();

    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()));



    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),vut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())));


    if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::svd(vut_);
	} else { //(!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::udt(vut_);
	}


    Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),uut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = ddt_(NSL::Slice(1,Nt),NSL::Ellipsis());
	Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vvt_(NSL::Slice(1,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    std::tie( Qnewtt_, Dnewtt_, Vnewtt_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewtt_, Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewtt_));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewtt_ ));

	corr_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()),invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()));


} // measureDiagonal()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureRow(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();


    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0 terms
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) + NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
    }

    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

/***
    // Nt/2<t<Nt
    vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),NSL::LinAlg::solve(PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())), false ));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));

    // 1<t<Nt/2 
    vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),PI_U_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),PI_V_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));



    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
	}



    // Nt/2<t<Nt
    invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
    invAp1F_Tt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), Qnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) ));

    // 1<t<Nt/2
    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve_triangular(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), false),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())));
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::adjoint(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()))); 
	}
    invAp1F_Tt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Qnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())), SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));
***/

    for (int t=1;t<Nt;t++){ // t>0 terms

      if((t+tsrc)%Nt > tsrc ) {
        corr_((t+tsrc)%Nt,NSL::Ellipsis()) = -NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
      } else {
        corr_((t+tsrc)%Nt,NSL::Ellipsis()) =  NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
      }

    }
    // do something and store in corr_

} // measureRow(tsrc)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureColumn(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();


    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis())+ NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis())),PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
    }


    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

/***
// 1<t<Nt/2
    vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())), false ));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));

    // Nt/2<t<Nt
    vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));



    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
	}
    

    
    // 1<t<Nt/2
    invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())));
    invAp1F_Tt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) ));

    // Nt/2<t<Nt
    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve_triangular(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), false),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()),NSL::LinAlg::adjoint(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()))),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	}
    invAp1F_Tt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Qnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())), PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));
***/

    for (int t=1;t<Nt;t++){ // t>0 terms

        if((t+tsrc)%Nt > tsrc){
            corr_((t+tsrc)%Nt,NSL::Ellipsis()) =  NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
        } else {
            corr_((t+tsrc)%Nt,NSL::Ellipsis()) = -NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
        }
     
    }


} // measureColumn(tsrc)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::calcPiSigma(NSL::size_t tsrc){

     NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
     NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
    std::tie(Uk_, expKdiag_, Vk_) = this->hfm_.Lat.svd_hopping( hfm_.delta_);  // note Uk.expKdiag.Vk = expK
    PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Vk_ * NSL::LinAlg::shift(this->hfm_.phiExp_,-tsrc).expand(Nx).transpose(1,2);
    PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag_*NSL::LinAlg::exp( hfm_.mu_);
    PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk_;

    SIGMA_V_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_D_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_U_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis());


    if (!this->hfm_.stabilityMethod.compare("SVD")) {
      for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(PI_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(PI_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(SIGMA_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(SIGMA_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
   } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
       for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(PI_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(PI_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(SIGMA_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(SIGMA_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
  }

} // calcPiSigma(tsrc)


} // namespace NSL::Measure::Hubbard

#endif // NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
