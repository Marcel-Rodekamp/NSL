#ifndef NSL_HMC_TPP
#define NSL_HMC_TPP

#include "Integrator/integrator.tpp"
#include "MarkovChain/markovState.tpp"
#include "Tensor/Factory/like.tpp"
#include "LinAlg.hpp"
#include "complex.hpp"
#include "concepts.hpp"
#include "IO.hpp"
#include "logger.hpp"
#include "stop.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace NSL::MCMC{

enum Chain{ AllStates, LastState };

template< NSL::Concept::isTemplateDerived<NSL::Integrator::Integrator> IntegratorType, 
          NSL::Concept::isTemplateDerived<NSL::Action::Action> ActionType
>
class HMC{
    public:

    HMC(const IntegratorType& integrator, const ActionType& action, NSL::H5IO & h5): 
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
    template<Chain chain, NSL::Concept::isNumber Type>
        std::conditional_t<chain == Chain::AllStates, std::vector<NSL::MCMC::MarkovState<Type>>, NSL::MCMC::MarkovState<Type>> 
    generate(const NSL::MCMC::MarkovState<Type> & state, NSL::size_t Nconf, NSL::size_t saveFrequency = 1, std::string baseNode = "markovChain"){
        // ensure that saveFrequency is at least 1. 
        if (saveFrequency <= 0) {
            saveFrequency = 1;
        }

        std::size_t logFrequency = 1;
        if(Nconf >= 100){
            logFrequency = static_cast<NSL::size_t>( 0.01*Nconf );
        }

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

            if (h5_.exist(fmt::format("{}/{}",baseNode,nstart))){
                // this reader looks for the most recent markov state and reads it into the state
                MC[nstart] = state;
                try {
                    h5_.read(MC[nstart], baseNode);
                }
                catch(...) {
                // in case the last markov state is corrupted try again
                    h5_.deleteData(fmt::format("{}/{}",baseNode,nstart));
                    nstart--;
                    MC[nstart] = state;
                    h5_.read(MC[nstart], baseNode, nstart);
                }

            } else{
                MC[nstart] = state;
                NSL::Logger::info("HMC: Starting new Markov Chain");
                h5_.write(MC[nstart],fmt::format("{}/{}",baseNode,nstart));
            }

            double runningAcceptance = 0;
            NSL::size_t backWindow = 100;

            // generate Nconf-1 configurations
            auto mc_time = NSL::Logger::start_profile("HMC");
            for(NSL::size_t n = nstart+1; n < Nconf; ++n){
                auto tmp = MC[n-1];
                
                // between each configuration generate saveFrequency which 
                // are not used for measurements
                for(NSL::size_t m = 0; m < saveFrequency-1; ++m){
                    tmp = this->generate_(tmp);
                }

                MC[n] = this->generate_(tmp);
		        h5_.write(MC[n],fmt::format("{}/{}",baseNode,n));

                runningAcceptance += static_cast<double>(MC[n].accepted);
                if ((n-backWindow) > 0) {
                    runningAcceptance -= static_cast<double>(MC[n-backWindow].accepted);
                }
                
                // ToDo: have a proper hook being called here
                if (n % logFrequency == 0){
                    NSL::Logger::info("HMC: {}/{}; Running Acceptance Rate: {:.6}%", n, Nconf, runningAcceptance/* *100. / backWindow (n-nstart) */ );
                    NSL::Logger::elapsed_profile(mc_time);
                }

                // safe early stop (NSL::Stop): configuration n is stored and flushed
                if (NSL::Stop::requested()) {
                    NSL::Logger::warn("HMC: stop requested ({}); configuration {} is stored. Run again to continue the chain.",
                        NSL::Stop::signalName(), n);
                    MC.resize(n+1);
                    break;
                }
            }
            NSL::Logger::stop_profile(mc_time);

            // return the Markov Chain
            return MC;
        } else {

            // for Chain::LastState we only need a new state that becomes overwritten over and over again.
            NSL::MCMC::MarkovState<Type> newState = state;


            // generate Nconf-1 configurations
            // As none is returned we just multiply the number of configurations
            completedSteps_ = 0;
            for(NSL::size_t n = 1; n < Nconf*saveFrequency; ++n){
                newState = this->generate_(newState);
                completedSteps_ = n;

                if (n % logFrequency == 0){
                    NSL::Logger::info("HMC: {}/{}", n, Nconf);
                }
                // safe early stop (NSL::Stop): the caller decides whether to store newState
                if (NSL::Stop::requested()) {
                    NSL::Logger::warn("HMC: stop requested ({}) after {} of {} steps.",
                        NSL::Stop::signalName(), n, Nconf*saveFrequency-1);
                    break;
                }
            }

            // return the Markov Chain
            return newState;
        }
    }

    template<Chain chain, NSL::Concept::isNumber Type>
        std::conditional_t<chain == Chain::AllStates, std::vector<NSL::MCMC::MarkovState<Type>>, NSL::MCMC::MarkovState<Type>> 
    generate(NSL::Configuration<Type> & config, NSL::size_t Nconf, NSL::size_t saveFrequency = 1){
        NSL::MCMC::MarkovState<Type> initialState(
            /*Configuration                        */ config,
            /*Action Value                         */ this->action_(config),
            /*Acceptance Probability               */ 1.,
            /*Markov Time                          */ 1 ,
            /*accepted                             */ true
        );
        return this->generate<chain,Type>(initialState, Nconf, saveFrequency);
    }

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

    //! Number of update steps done by the last generate<Chain::LastState> call:
    //! Nconf*saveFrequency-1, or fewer if it stopped early on a signal (see NSL::Stop).
    NSL::size_t completedSteps() const { return completedSteps_; }

    //! A trajectory that throws or yields a NaN energy is rejected and logged; after this
    //! many failures in a row (default 10) the run is aborted with an exception.
    void setMaxConsecutiveFailures(NSL::size_t n) { maxConsecutiveFailures_ = n; }

    protected:

    //! Implementation of the HMC
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> generate_(const NSL::MCMC::MarkovState<Type> & state){

        // sample momentum (unit mass unless the integrator defines a mass matrix)
        NSL::Configuration<Type> momentum = NSL::Integrator::sampleMomentum(integrator_, state.configuration);

        std::string failure;
        try {
            // use integrator to generate proposal
            auto [proposal_config,proposal_momentum] = this->integrator_(state.configuration,momentum);

            // compute the Action
            Type proposal_S = this->action_(proposal_config);

            // compute the Hamiltonian H = K(p) + S, K(p) = p^2/2 for unit mass
            // Starting point of the trajectory
            Type starting_H = state.actionValue;
            starting_H += NSL::Integrator::kineticEnergy(integrator_, momentum);

            // End point of the trajectory i.e. proposal
            Type proposal_H = proposal_S;
            proposal_H += NSL::Integrator::kineticEnergy(integrator_, proposal_momentum);

            // We always assume real part of the action, i.e. automatic reweighting
            // for complex actions!
            const NSL::RealTypeOf<Type> logAcceptance = NSL::real(starting_H - proposal_H);
            const double r = r_.rand()[0];

            // A non-finite energy difference is a rejection with probability 0, also when
            // e^{-dH} would be infinite (an overflow in the proposal must never be accepted).
            // +-inf is a diverged trajectory; NaN is a numerical breakdown and counts as a failure.
            if (std::isnan(logAcceptance)) {
                failure = fmt::format("the energy difference is NaN (H at the start: {}, at the end: {})",
                                      NSL::real(starting_H), NSL::real(proposal_H));
            } else if (!std::isfinite(logAcceptance)) {
                failures_ = 0;
                return rejected_(state);
            } else {
                failures_ = 0;
                const NSL::RealTypeOf<Type> acceptanceProb = NSL::LinAlg::exp(logAcceptance);

                // accept reject
                if ( r <= acceptanceProb ){
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
                        acceptanceProb,
                        state.markovTime+1,
                        false
                    );
                }
            }
        } catch (const std::exception & e) {
            failure = e.what();
        } catch (...) {
            failure = "unknown exception";
        }

        // a failed trajectory is rejected and logged; too many in a row abort the run
        ++failures_;
        NSL::Logger::warn("HMC trajectory failed and is rejected ({} in a row): {}", failures_, failure);
        if (failures_ >= maxConsecutiveFailures_) {
            throw std::runtime_error(fmt::format(
                "HMC trajectory failed {} times in a row; last error: {}", failures_, failure));
        }
        return rejected_(state);
    }

    //! The state after a rejected (or failed) update: acceptance probability 0
    template<NSL::Concept::isNumber Type>
    NSL::MCMC::MarkovState<Type> rejected_(const NSL::MCMC::MarkovState<Type> & state) const {
        return NSL::MCMC::MarkovState<Type>(
            state.configuration,
            state.actionValue,
            NSL::RealTypeOf<Type>(0),
            state.markovTime+1,
            false
        );
    }

    private:
    //! ToDo: We need to implement a proper RNG class!
    NSL::Tensor<double> r_;

    IntegratorType integrator_;
    ActionType action_;

    NSL::H5IO h5_;

    //! update steps of the last generate<Chain::LastState> call
    NSL::size_t completedSteps_ = 0;
    //! failed trajectories in a row (exception or NaN)
    NSL::size_t failures_ = 0;
    NSL::size_t maxConsecutiveFailures_ = 10;

}; // HMC

} // namespace NSL::MCMC

#endif //NSL_HMC_TPP
