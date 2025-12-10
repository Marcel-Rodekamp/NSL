#ifndef NSL_FERMION_PROPAGATOR_TPP
#define NSL_FERMION_PROPAGATOR_TPP

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
class FermionPropagator: public Measurement {
    public:
        FermionPropagator(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            corr_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            phi_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            basenode_(basenode_),
	    invAp1F_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	invAp1F_U_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	invAp1F_D_(lattice.device(), 1, lattice.sites()),
	invAp1F_T_(lattice.device(), 1, lattice.sites(), lattice.sites()),
        pi_dot_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites()),
	expKdiag_(lattice.device(), lattice.sites()),
	Uk_(lattice.device(), lattice.sites(), lattice.sites()),
	Vk_(lattice.device(), lattice.sites(), lattice.sites()),
	PI_U_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
        PI_D_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites()),
        PI_V_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
        SIGMA_U_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
        SIGMA_D_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites()),
        SIGMA_V_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
        Fk_U_(lattice.device(),lattice.sites(),lattice.sites()),
        Fk_D_(lattice.device(),lattice.sites()),   
	Fk_V_(lattice.device(),lattice.sites(),lattice.sites()),
	uu_(lattice.device(), lattice.sites(), lattice.sites()),
	dd_(lattice.device(), lattice.sites()),
	vv_(lattice.device(), lattice.sites(), lattice.sites()),
	vu_(lattice.device(), lattice.sites(), lattice.sites()),
	udv_(lattice.device(), lattice.sites(), lattice.sites()),
	Qnew_(lattice.device(), lattice.sites(), lattice.sites()),
	Dnew_(lattice.device(), lattice.sites()),
	Vnew_(lattice.device(), lattice.sites(), lattice.sites())
    {}

    FermionPropagator(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
        FermionPropagator(
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
    
    void measureDiagonal();
    void measureRow();
    void measureColumn();

    void calcPiSigma(NSL::size_t tsrc);

    // maybe make these private later. . .
    NSL::Tensor<Type> invAp1F_;
    NSL::Tensor<Type> pi_dot_;
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
    NSL::Tensor<Type> udv_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnew_; //(device, Nx);
    NSL::Tensor<Type> Vnew_; //(device, Nx, Nx);

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
    NSL::Hubbard::Species species_;

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> phi_;

    std::string basenode_;
};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID){

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

        // first do particle species
	species_ = NSL::Hubbard::Particle;

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/particle/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,species_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn();
	   //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}
	
	// calculate row
	node = fmt::format("/markovChain/{}/propagator/particle/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,species_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow();
	   //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/particle/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,species_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	  }
	}

	PiSigma = true; // reset bool for the holes. . .
	
	// now repeat for hole species
	species_ = NSL::Hubbard::Hole;
		
	// populate the fermion matrix using the free configuration
    	hfm_.populate(phi_,species_);

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/hole/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,species_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn();
	   //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}

	// calculate row
	node = fmt::format("/markovChain/{}/propagator/hole/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,species_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow();
	   //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/hole/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,species_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      //this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	   }
      }

    } // tsrc
      
} // measure(Ntsrc,config);

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::FermionPropagator");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator 
    std::string node;
/*    if (species_ == NSL::Hubbard::Particle){
          node = "/NonInteracting/propagator/particle/invM[t,t]";
    } else {
          node = "/NonInteracting/propagator/hole/invM[t,t]";
    }
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
    
        // this stores the result in corr_
        //measure(1);

        // write the calculated correlator to file
       h5_.write(corr_,std::string(basenode_)+node);
    } else {
        NSL::Logger::info("Non-interacting invM[t,t] already exists");
    }
*/

    // need to finish calculating non-interacting propagator for rows and columns!!!


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
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureDiagonal(){

    // do something and store in corr_

} // measureDiagonal()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureRow(){

    // do something and store in corr_

} // measureRow()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureColumn(){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();

    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
    invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew_;
    invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(
      		                      NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),
				          invAp1F_T_(0,NSL::Ellipsis()));

    for (int t=1;t<Nt;t++){ // t>0 terms
      if (t<=Nt/2){

      } else {

      }

    }


} // measureColumn()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::calcPiSigma(NSL::size_t tsrc){

     NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
     NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
    std::tie(Uk_, expKdiag_, Vk_) = this->hfm_.Lat.svd_hopping(hfm_.sgn_* hfm_.delta_);  // note Uk.expKdiag.Vk = expK
    PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Vk_ * NSL::LinAlg::shift(this->hfm_.phiExp_,-tsrc).expand(Nx).transpose(1,2);
    PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag_*NSL::LinAlg::exp(hfm_.sgn_*hfm_.mu_);
    PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk_;

    SIGMA_V_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_D_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_U_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis());


    if (!this->hfm_.stabilityMethod.compare("QR")) {
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
   } else if (!this->hfm_.stabilityMethod.compare("SVD")) {
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
  }

} // calcPiSigma(tsrc)

/*
if (!this->stabilityMethod.compare("QR")) {

      // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
      std::tie(Uk_, expKdiag_, Vk_) = this->Lat.svd_hopping(sgn_* delta_);  // note Uk.expKdiag.Vk = expK
      Fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Vk_ * NSL::LinAlg::shift(this->phiExp_,+1).expand(Nx).transpose(1,2);
      Fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag_*NSL::LinAlg::exp(sgn_*this->mu_);
      Fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk_;

      fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis());
      fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis());
      fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis());

      for (int t=1;t<Nt;t++) {
      	  // Fkt(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(Fkt_V_(t,NSL::Ellipsis()),Fkt_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(Fkt_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(Fkt_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
      	  Fkt_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(Fkt_U_(t,NSL::Ellipsis()),uu_);
	  Fkt_D_(t,NSL::Ellipsis()) = dd_;
	  Fkt_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,Fkt_V_(t-1,NSL::Ellipsis()));

	  // fk(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(fkt_V_(Nt-t,NSL::Ellipsis()),fkt_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(fkt_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(fkt_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
	  fkt_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(fkt_U_(Nt-t,NSL::Ellipsis()),uu_);
	  fkt_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  fkt_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,fkt_V_(Nt-1-t,NSL::Ellipsis()));
      }

      // calculation of forces
      for(int t=0;t<Nt-1;t++) {

          vu_ = NSL::LinAlg::mat_mul(Fkt_V_(t,NSL::Ellipsis()),fkt_U_(t+1,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(Fkt_D_(t,NSL::Ellipsis())),vu_);
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(fkt_D_(t+1,NSL::Ellipsis())));
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
      	  Fk_U_(NSL::Ellipsis()) = NSL::LinAlg::mat_mul(Fkt_U_(t,NSL::Ellipsis()),uu_);
	  Fk_D_(NSL::Ellipsis()) = dd_;
	  Fk_V_(NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,fkt_V_(t+1,NSL::Ellipsis()));

	  std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fk_V_(NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_U_(NSL::Ellipsis())),false ) + NSL::LinAlg::diag(Fk_D_(NSL::Ellipsis())));

          invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, Fk_V_(NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
      	  invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew_;
      	  invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fk_U_(NSL::Ellipsis()), Qnew_ ));

	  pi_dot_(t,NSL::Slice()) = II * NSL::LinAlg::diag(
	                                 NSL::LinAlg::mat_mul(
      		                         NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),
				         invAp1F_T_(0,NSL::Ellipsis())));

      }
      // Nt-1 timeslice (last timeslice)
      std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fkt_V_(Nt-1,NSL::Ellipsis()), NSL::LinAlg::adjoint(Fkt_U_(Nt-1,NSL::Ellipsis())),false ) + NSL::LinAlg::diag(Fkt_D_(Nt-1,NSL::Ellipsis())));

      invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, Fkt_V_(Nt-1,NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
      invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew_;
      invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U_(Nt-1,NSL::Ellipsis()), Qnew_ ));

      pi_dot_(Nt-1,NSL::Slice()) = II * NSL::LinAlg::diag(
	                             NSL::LinAlg::mat_mul(
      		                     NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),
				         invAp1F_T_(0,NSL::Ellipsis())));

      return pi_dot_;	

    }
    else if (!this->stabilityMethod.compare("SVD")) {

      // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
      std::tie(Uk_, expKdiag_, Vk_) = this->Lat.svd_hopping(sgn_* delta_);  // note Uk.expKdiag.Vk = expK
      Fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Vk_ * NSL::LinAlg::shift(this->phiExp_,+1).expand(Nx).transpose(1,2);
      Fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag_*NSL::LinAlg::exp(sgn_*this->mu_);
      Fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk_;

      fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_V_(NSL::Slice(0,Nt),NSL::Ellipsis());
      fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_D_(NSL::Slice(0,Nt),NSL::Ellipsis());
      fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis())=Fkt_U_(NSL::Slice(0,Nt),NSL::Ellipsis());

      for (int t=1;t<Nt;t++) {
      	  // Fkt(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(Fkt_V_(t,NSL::Ellipsis()),Fkt_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(Fkt_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(Fkt_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
      	  Fkt_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(Fkt_U_(t,NSL::Ellipsis()),uu_);
	  Fkt_D_(t,NSL::Ellipsis()) = dd_;
	  Fkt_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,Fkt_V_(t-1,NSL::Ellipsis()));

	  // fk(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(fkt_V_(Nt-t,NSL::Ellipsis()),fkt_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(fkt_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(fkt_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
	  fkt_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(fkt_U_(Nt-t,NSL::Ellipsis()),uu_);
	  fkt_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  fkt_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,fkt_V_(Nt-1-t,NSL::Ellipsis()));
      }

      // calculation of forces
      for(int t=0;t<Nt-1;t++) {

          vu_ = NSL::LinAlg::mat_mul(Fkt_V_(t,NSL::Ellipsis()),fkt_U_(t+1,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(Fkt_D_(t,NSL::Ellipsis())),vu_);
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(fkt_D_(t+1,NSL::Ellipsis())));
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
      	  Fk_U_(NSL::Ellipsis()) = NSL::LinAlg::mat_mul(Fkt_U_(t,NSL::Ellipsis()),uu_);
	  Fk_D_(NSL::Ellipsis()) = dd_;
	  Fk_V_(NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,fkt_V_(t+1,NSL::Ellipsis()));

          std::tie( Qnew_ , Dnew_ , Vnew_ ) = NSL::LinAlg::udt(
      		NSL::LinAlg::solve(Fk_V_(NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_U_(NSL::Ellipsis())), false )
		+ NSL::LinAlg::diag(Fk_D_(NSL::Ellipsis())) );  // Note: I still use QR to stabilize the inverse, even though this is the "SVD" stabilizer routine

	  invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, Fk_V_(NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
          invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew_;
          invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fk_U_(NSL::Ellipsis()), Qnew_ ));

	  pi_dot_(t,NSL::Slice()) = II * NSL::LinAlg::diag(
	                                 NSL::LinAlg::mat_mul(
      		                         NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),
				         invAp1F_T_(0,NSL::Ellipsis())));

      }
      // Nt-1 timeslice (last timeslice)
      std::tie( Qnew_ , Dnew_ , Vnew_ ) = NSL::LinAlg::udt(
      		NSL::LinAlg::solve(Fkt_V_(Nt-1,NSL::Ellipsis()), NSL::LinAlg::adjoint(Fkt_U_(Nt-1,NSL::Ellipsis())), false )
		+ NSL::LinAlg::diag(Fkt_D_(Nt-1,NSL::Ellipsis())) );  // Note: I still use QR to stabilize the inverse, even though this is the "SVD" stabilizer routine

      invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, Fkt_V_(Nt-1,NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
      invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew_;
      invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U_(Nt-1,NSL::Ellipsis()), Qnew_ ));

      pi_dot_(Nt-1,NSL::Slice()) = II * NSL::LinAlg::diag(
	                             NSL::LinAlg::mat_mul(
      		                     NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),
				         invAp1F_T_(0,NSL::Ellipsis())));

      return pi_dot_;

    }
*/


} // namespace NSL::Measure::Hubbard

#endif // NSL_FERMION_PROPAGATOR_TPP
