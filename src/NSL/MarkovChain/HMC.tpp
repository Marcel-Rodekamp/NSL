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

            if (nstart > 0){
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
                    NSL::Logger::info("HMC: {}/{}; Running Acceptence Rate: {:.6}%", n, Nconf, runningAcceptance/* *100. / backWindow (n-nstart) */ );
                    NSL::Logger::elapsed_profile(mc_time);
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
                acceptanceProb,
                state.markovTime+1,
                false
            );
        }
    }

    private:
    //! ToDo: We need to implement a proper RNG class!
    NSL::Tensor<double> r_;

    IntegratorType integrator_;
    ActionType action_;

    NSL::H5IO h5_;

}; // HMC

} // namespace NSL::MCMC

#endif //NSL_HMC_TPP
