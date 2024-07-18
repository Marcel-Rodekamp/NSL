#ifndef NSL_RadialHMC_TPP
#define NSL_RadialHMC_TPP

#include "Integrator/integrator.tpp"
#include "MarkovChain/markovState.tpp"
// #include "MarkovChain/radialMarkovState.tpp"
#include "Tensor/Factory/like.tpp"
#include "LinAlg.hpp"
#include "complex.hpp"
#include "concepts.hpp"
#include "IO.hpp"
#include "logger.hpp"
#include <fstream>

namespace NSL::MCMC{

// enum Chain{ AllStates, LastState }; // FT: what is this? Do I need this here again if it is already in HMC.tpp

template< NSL::Concept::isTemplateDerived<NSL::Integrator::Integrator> IntegratorType, 
          NSL::Concept::isTemplateDerived<NSL::Action::Action> ActionType
>
class RadialHMC{
    public:

    RadialHMC(const IntegratorType& integrator, const ActionType& action, NSL::H5IO & h5): 
        r_(1),
        integrator_(integrator),
        action_(action),
	    h5_(h5)
    {}

    //! Generate a single Markov Chain element from the input state 
    /*!
     * This implementation calls the Hybrid/Hamilton Monte Carlo (HMC) algorithm.
     * */
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> generate(const NSL::MCMC::MarkovState<Type> & state){
        return this->generate_<Type>(state);
    }

    //! Generate a Markov Chain (element)
    /*! 
     * \p state, Initial markov chain element
     * \p Nconf, Number of configurations to be generated
     * \p saveFrequency, Number of configurations to be skipped generated
     *                  (default=1)
     *
     * if chain == Chain::AllStates:
     *      Returns a `std::vector<MarkovState<Type>>` of size \f(N_{conf}\f) 
     *      representing the Markov Chain where only every `saveFrequency`s 
     *      element is returned
     *      Usage Example: Production of independent Markov Chain
     * else:
     *      Returns a `MarkovState<Type>` produced as the \f(N_{conf}\f)s 
     *      element of the Markov Chain
     *      Usage Example: Burn In
     *
     * */

    /* 
        - generate with radial update step: 
        - TODO: resolve ambiguity with other generate due to default args (resolved by radialScale?)
        - TODO: radialScale/Type can be complex? has to be cast to real if that is the case
    */


    template<Chain chain, NSL::Concept::isNumber Type>
        std::conditional_t<chain == Chain::AllStates, std::vector<NSL::MCMC::MarkovState<Type>>, NSL::MCMC::MarkovState<Type>> 
    generate(NSL::MCMC::MarkovState<Type> & state, NSL::size_t Nconf, NSL::size_t radialFrequency,  NSL::RealTypeOf<Type> radialLoc, NSL::RealTypeOf<Type> radialScale, std::string racc_file, NSL::size_t saveFrequency = 1, std::string baseNode = "markovChain"){
        // ensure that saveFrequency is at least 1. 
        if (saveFrequency <= 0) {
            saveFrequency = 1;
        }

        // ensure that radialFrequency is non-negative. In case it is, it's set s.t. no radial updates occur
        if (radialFrequency <= 0) {
            radialFrequency = saveFrequency * Nconf + 1;
        }

        std::size_t logFrequency = 1;
        if(Nconf >= 100){
            logFrequency = static_cast<NSL::size_t>( 0.01*Nconf );
        }

        NSL::RealTypeOf<Type> volume = state.configuration["phi"].numel(); // volume, i.e. number of total sites for accept reject in radial update step
        int N_radial_steps = saveFrequency * Nconf / radialFrequency;
        std::vector<int> racc_hist(N_radial_steps);
        std::cout << N_radial_steps << std::endl;
        int rhist_count = 0;
        if constexpr(chain == Chain::AllStates) {
            // prepare some memory to all states
            std::vector<NSL::MCMC::MarkovState<Type>> MC(Nconf);

            // Put the initial configuration in the 0th element
            
            // check if the file already contains configs and if we are allowed to overwrite them
            NSL::size_t nstart = 0;
            
            if (h5_.exist(baseNode)){
                auto [minConfigID, maxConfigID] = h5_.getMinMaxConfigs(baseNode);
                nstart = maxConfigID;
            }

            if (nstart > 0){
                // this reader looks for the most recent markov state and reads it into the state
                MC[nstart] = state;
                h5_.read(MC[nstart], baseNode);

            } else{
                MC[nstart] = state;
                NSL::Logger::info("HMC: Starting new Markov Chain");
                h5_.write(MC[nstart],fmt::format("{}/{}",baseNode,nstart));
            }

            double runningAcceptance = 0;
            double runningRadialAcceptance = 0;
            double increasedRadiusAcceptance = 0;
            double decreasedRadiusAcceptance = 0;
            double proposed_increase_count = 0;
            double summed_accepted_incr_radius = 0;
            double summed_accepted_decr_radius = 0;
            int step_count = 0; // counts total steps to keep track of when to perform a radial update
            int rstep_count = 0; // counts radial update steps to compute radial acceptance rate
            // generate Nconf-1 configurations
            auto mc_time = NSL::Logger::start_profile("HMC");
            for(NSL::size_t n = nstart+1; n < Nconf; ++n){
                auto tmp = MC[n-1];
                
                // between each configuration generate saveFrequency which 
                // are not used for measurements
                for(NSL::size_t m = 0; m < saveFrequency-1; ++m){
                    step_count++;
                    // Check for radial update step
                    if (step_count % radialFrequency == 0){
                        rstep_count++;
                        tmp = this->radialgenerate_(tmp, volume, radialLoc, radialScale);
                        runningRadialAcceptance += static_cast<double>(tmp.accepted);
                        racc_hist[rhist_count] = static_cast<int>(tmp.accepted);
                        rhist_count += 1;
                        // out_racc << static_cast<double>(tmp.accepted) << "\n";
                        // Record whether radius was increased and whether this was accepted
                        if (bool_rincr_){
                            proposed_increase_count += 1;
                            increasedRadiusAcceptance += static_cast<double>(tmp.accepted);
                            if (tmp.accepted) {
                                summed_accepted_incr_radius += 1-1/curr_rscale_;
                            }
                        }
                        else{
                            decreasedRadiusAcceptance += static_cast<double>(tmp.accepted);
                            if (tmp.accepted) {
                                summed_accepted_decr_radius += 1-curr_rscale_;
                            }
                        }
                        if (rstep_count % logFrequency == 0){
                            NSL::Logger::info("HMC: {}/{}; Running Radial Acceptance Rate {:.6}%", n, Nconf, runningRadialAcceptance*100./rstep_count);
                        }
                    }
                    tmp = this->generate_(tmp);
                }

                // Check for radial update step
                step_count++;
                if (step_count % radialFrequency == 0){
                    rstep_count++;
                    tmp = this->radialgenerate_(tmp, volume, radialLoc, radialScale);
                    runningRadialAcceptance += static_cast<double>(tmp.accepted);
                    racc_hist[rhist_count] = static_cast<int>(tmp.accepted);
                    rhist_count += 1;
                    // out_racc << static_cast<double>(tmp.accepted) << "\n";
                    if (bool_rincr_){
                        proposed_increase_count += 1;
                        increasedRadiusAcceptance += static_cast<double>(tmp.accepted);
                        if (tmp.accepted) {
                            summed_accepted_incr_radius += 1-1/curr_rscale_;
                        }
                    }
                    else{
                        decreasedRadiusAcceptance += static_cast<double>(tmp.accepted);
                        if (tmp.accepted) {
                            summed_accepted_decr_radius += 1-curr_rscale_;
                        }
                    }
                    if (rstep_count % logFrequency == 0){
                        NSL::Logger::info("HMC: {}/{}; Running Radial Acceptance Rate {:.6}%", n, Nconf, runningRadialAcceptance*100./rstep_count);
                    } 
                }
                MC[n] = this->generate_(tmp);
		        h5_.write(MC[n],fmt::format("{}/{}",baseNode,n));

                runningAcceptance += static_cast<double>(MC[n].accepted);

                // ToDo: have a proper hook being called here
                if (n % logFrequency == 0){
                    NSL::Logger::info("HMC: {}/{}; Running Acceptence Rate: {:.6}%", n, Nconf, runningAcceptance*100./(n-nstart));
                    //NSL::Logger::info("HMC: {}/{}; Running Acceptence Rate: {:.6}%; Running Radial Acceptance Rate {:.6}%", n, Nconf, runningAcceptance*100./(n-nstart), runningRadialAcceptance*100./rstep_count);
                    NSL::Logger::elapsed_profile(mc_time);
                }
            }
            // Logging of information on radial Metropolis updates
            double incr_acc = increasedRadiusAcceptance/proposed_increase_count;
            double decr_acc = decreasedRadiusAcceptance/(rstep_count-proposed_increase_count);

            std::ofstream out_racc{ racc_file };

            for (int i = 0; i<racc_hist.size(); ++i) {
                out_racc << racc_hist[i] << "\n";
            }
            out_racc.close();


            std::cout << "Version 4" << std::endl;
            NSL::Logger::info("proposed_increase_count: {:.6}", (proposed_increase_count));
            NSL::Logger::info("summed accepted incr radius: {:.6}", (summed_accepted_incr_radius));
            NSL::Logger::info("summed accepted decr radius: {:.6}", (summed_accepted_decr_radius));
            NSL::Logger::info("rstep_count: {:.6}", (static_cast<double>(rstep_count)));
            NSL::Logger::info("incr acc: {:.6}", (incr_acc));
            NSL::Logger::info("decr acc: {:.6}", (decr_acc));


            NSL::Logger::info("Increase in Radius Proposed: {:.6}%", (100.*proposed_increase_count/rstep_count));
            NSL::Logger::info("Increase in Radius Accepted: {:.6}%", (100.*incr_acc));
            NSL::Logger::info("Decrease in Radius Proposed: {:.6}%", (100.*(1-proposed_increase_count/rstep_count)));
            NSL::Logger::info("Decrease in Radius Accepted: {:.6}%", (100.*decr_acc));
            NSL::Logger::info("Measure of increased exploration: {:.10}", (summed_accepted_incr_radius/proposed_increase_count));
            NSL::Logger::info("Measure of decreased exploration: {:.10}", (summed_accepted_decr_radius/(rstep_count-proposed_increase_count)));
            NSL::Logger::stop_profile(mc_time);

            // return the Markov Chain
            return MC;
        } else {

            // for Chain::LastState we only need a new state that becomes overwritten over and over again.
            NSL::MCMC::MarkovState<Type> newState = state;


            // generate Nconf-1 configurations
            // As none is returned we just multiply the number of configurations
            for(NSL::size_t n = 1; n < Nconf*saveFrequency; ++n){
                newState = this->generate_(newState);

                if (n % logFrequency == 0){
                    NSL::Logger::info("HMC: {}/{}", n, Nconf);
                }
            }

            // return the Markov Chain
            return newState;
        }
    }

    template<Chain chain, NSL::Concept::isNumber Type>
        std::conditional_t<chain == Chain::AllStates, std::vector<NSL::MCMC::MarkovState<Type>>, NSL::MCMC::MarkovState<Type>> 
    generate(NSL::Configuration<Type> & config, NSL::size_t Nconf, NSL::size_t radialFrequency,  NSL::RealTypeOf<Type> radialLoc, NSL::RealTypeOf<Type> radialScale, std::string racc_file, NSL::size_t saveFrequency = 1){
        NSL::MCMC::MarkovState<Type> initialState(
            /*Configuration                        */ config,
            /*Action Value                         */ this->action_(config),
            /*Acceptance Probability               */ 1.,
            /*Markov Time                          */ 1 ,
            /*accepted                             */ true
        );
        return this->generate<chain,Type>(initialState, Nconf, radialFrequency, radialLoc, radialScale, racc_file, saveFrequency);
    }

    // a single pure HMC update
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> generate(NSL::Configuration<Type> & config){
        NSL::MCMC::MarkovState<Type> initialState(
            /*Configuration                        */ config,
            /*Action Value                         */ this->action_(config),
            /*Acceptance Probability               */ 1.,
            /*Markov Time                          */ 1 ,
            /*accepted                             */ true
        );
        return this->generate_<Type>(initialState);
    }

    // a single radial HMC update
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> generate(NSL::Configuration<Type> & config, NSL::size_t radialLoc, NSL::RealTypeOf<Type> radialScale){
        NSL::MCMC::MarkovState<Type> initialState(
            /*Configuration                        */ config,
            /*Action Value                         */ this->action_(config),
            /*Acceptance Probability               */ 1.,
            /*Markov Time                          */ 1 ,
            /*accepted                             */ true
        );
        NSL::RealTypeOf<Type> volume = config["phi"].numel(); // volume, i.e. number of total sites for accept reject in radial update step
        return this->radialgenerate_<Type>(initialState, volume, radialLoc, radialScale);
    }

    

    protected:

    //! Implementation of the HMC
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> generate_(const NSL::MCMC::MarkovState<Type> & state){

        // sample momentum 
        NSL::Configuration<Type> momentum;
        for(auto & [key,field]: state.configuration){
            NSL::Tensor<Type> p = NSL::zeros_like(field);
            p.randn();
	        p.imag()=0;
            momentum[key] = p; 
        }

        // use integrator to generate proposal 
         auto [proposal_config,proposal_momentum] = this->integrator_(state.configuration,momentum);

        // compute the Action
        Type proposal_S = this->action_(proposal_config);

        // compute the Hamiltonian H = p^2/2 + S
        // Starting point of the trajectory
        Type starting_H = state.actionValue;
        for( const auto& [key,field]: momentum){
            starting_H += 0.5*(field * field).sum();
        }

        
        // End point of the trajectory i.e. proposal
        Type proposal_H = proposal_S;
        for( const auto& [key,field]: proposal_momentum){
            proposal_H += 0.5*(field * field).sum();
        }

        // We always assume real part of the action, i.e. automatic reweighting
        // for complex actions!
        NSL::RealTypeOf<Type> acceptanceProb = NSL::LinAlg::exp( NSL::real(starting_H - proposal_H) );


        // accept reject
	    if ( r_.rand()[0] <= acceptanceProb ){
            return NSL::MCMC::MarkovState<Type>{
                proposal_config,
                proposal_S,
                acceptanceProb,
                state.markovTime+1,
                true
                /*For this algorithm there are no weights to be added*/
            };
        } else {
            return NSL::MCMC::MarkovState<Type>(
                state.configuration,
                state.actionValue,
                state.acceptanceProbability,
                state.markovTime+1,
                false
            );
        }
    }

    //! Implementation of the HMC radial update
    template<NSL::Concept::isNumber Type>
    // NSL::MCMC::MarkovState<Type> radialgenerate_(const NSL::MCMC::MarkovState<Type> & state,  NSL::RealTypeOf<Type> volume, NSL::RealTypeOf<Type> radialLoc, NSL::RealTypeOf<Type> radialScale){
    NSL::MCMC::MarkovState<Type> radialgenerate_(const NSL::MCMC::MarkovState<Type> & state, NSL::RealTypeOf<Type> volume, NSL::RealTypeOf<Type> radialLoc, NSL::RealTypeOf<Type> radialScale){
        NSL::Tensor<NSL::RealTypeOf<Type>> rscaleTensor_(tmp_size_);
        rscaleTensor_.lognormal(radialLoc, radialScale);  // FT: completely remove tensor?
        NSL::RealTypeOf<Type> rscale_ = NSL::real(rscaleTensor_[0]);
        // NSL::RealTypeOf<Type> inv_rscale_ = 1/rscale_;
        bool_rincr_ = (rscale_ >= 1); // tmp bool for recording whether radius was increased
        curr_rscale_ = static_cast<double>(rscale_);
        // compute the Action
        NSL::Configuration<Type> proposal_config (state.configuration,true);
        // proposal_config *= rscale_[0]; //static_cast<Type> (rscale_[0]); // Need elementwise tensor operation or static_cast here?
        proposal_config = static_cast<Type>(rscale_) * proposal_config; //FT : Why can't I use "*=" here?

        Type proposal_S = this->action_(proposal_config);

        // Starting point of the trajectory
        Type starting_S = state.actionValue;

        // We always assume real part of the action, i.e. automatic reweighting
        // for complex actions!

        // NSL::RealTypeOf<Type> acceptanceProb = NSL::LinAlg::exp( NSL::real(starting_S - proposal_S) ) * prob_inv_rscale[0]/prob_rscale[0];
        NSL::RealTypeOf<Type> acceptanceProb = NSL::LinAlg::exp( NSL::real(starting_S - proposal_S) + volume * NSL::LinAlg::log(rscale_)) ;

        // accept reject // FT: markovTime+1 in radial step? Different accepted to get separate acceptance rate for radial updates?
	    if ( r_.rand()[0] <= acceptanceProb ){
            return NSL::MCMC::MarkovState<Type>{
                proposal_config,
                proposal_S,
                acceptanceProb,
                state.markovTime+1,
                true
                /*For this algorithm there are no weights to be added*/
            };
        } else {
            return NSL::MCMC::MarkovState<Type>(
                state.configuration,
                state.actionValue,
                state.acceptanceProbability,
                state.markovTime+1,
                false
            );
        }
    }


    template<NSL::Concept::isNumber RealType>
    RealType lognormalProb_(RealType val, RealType mean, RealType std){
        RealType prob = NSL::LinAlg::exp( - 0.5*(NSL::LinAlg::log(val) - mean) * (NSL::LinAlg::log(val) - mean)/(std*std) ) / ( val * std * 2.5066282746310002);
        return prob;
    }

    private:
    //! ToDo: We need to implement a proper RNG class!
    NSL::Tensor<double> r_;
    NSL::size_t tmp_size_ = 1;
    bool bool_rincr_ = false; 
    double curr_rscale_;

    IntegratorType integrator_;
    ActionType action_;

    NSL::H5IO h5_;

}; // RadialHMC

} // namespace NSL::MCMC

#endif //NSL_HMC_TPP
