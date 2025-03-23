#ifndef NSL_MARKOV_STATE_TPP
#define NSL_MARKOV_STATE_TPP

#include <concepts>
#include <type_traits>
#include "Configuration/Configuration.tpp"
#include "complex.hpp"
#include "types.hpp"

namespace NSL::MCMC {


template<NSL::Concept::isNumber Type>
class MarkovState{
    public:

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    MarkovState(
        NSL::Configuration<Type> config,
        const Type & actionValue,
        const NSL::RealTypeOf<Type> & acceptanceProbability,
        NSL::size_t markovTime,
        bool accepted,
        WeightType ... weights
    ): 
        configuration(config),
        actionValue(actionValue),
        weights({weights...}),
        acceptanceProbability(acceptanceProbability),
        markovTime(markovTime),
        accepted(accepted)
    {}

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    MarkovState(
        NSL::Configuration<Type> config,
        const Type & actionValue,
        const NSL::RealTypeOf<Type> & acceptanceProbability,
        WeightType ... weights
    ): 
        configuration(config),
        actionValue(actionValue),
        weights({weights...}),
        acceptanceProbability(acceptanceProbability),
        markovTime(1),
        accepted(1)
    {}

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    MarkovState(
        NSL::Configuration<Type> config,
        NSL::Configuration<Type> pseudoFermion,
        const Type & actionValue,
        const NSL::RealTypeOf<Type> & acceptanceProbability,
        NSL::size_t markovTime,
        bool accepted,
        WeightType ... weights
    ): 
        configuration(config),
        pseudoFermion(pseudoFermion),
        actionValue(actionValue),
        weights({weights...}),
        acceptanceProbability(acceptanceProbability),
        markovTime(markovTime),
        accepted(accepted)
    {}

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    MarkovState(
        NSL::Configuration<Type> config,
        NSL::Configuration<Type> pseudoFermion,
        const Type & actionValue,
        const NSL::RealTypeOf<Type> & acceptanceProbability,
        WeightType ... weights
    ): 
        configuration(config),
        pseudoFermion(pseudoFermion),
        actionValue(actionValue),
        weights({weights...}),
        acceptanceProbability(acceptanceProbability),
        markovTime(1),
        accepted(1)
    {}

    MarkovState() = default;

    MarkovState( const MarkovState<Type> & ) = default;
    MarkovState( MarkovState<Type> && ) = default;

    MarkovState<Type>& operator=(const MarkovState<Type>&) = default;
    MarkovState<Type>& operator=(MarkovState<Type>&&) = default;

    //! Store the configuration associated with the Markov State
    NSL::Configuration<Type> configuration;

    //! Store pseudo fermion fields if required
    NSL::Configuration<Type> pseudoFermion;

    //! Store the associated action value
    Type actionValue;

    //! Store additional weights for the measure of required (otherwise leave empty map)
    std::map<std::string, Type> weights;

    //! Store the probability with which it was accepted
    NSL::RealTypeOf<Type> acceptanceProbability;

    // ToDo:
    //! Store the RNG State
    //std::pair<int,int,int> RNGState;
    
    // ToDo:
    //! Store the number of threads used to generate this MC-State
    //int numberThreads;
    
    //! Store Markov Time, an integer referencing the Markov State in the 
    //! Markov Chain
    NSL::size_t markovTime;

    //! Store if this configuration was accepted from the previous step
    //! true : 1
    //! false: 0
    bool accepted;
};

template<NSL::Concept::isNumber Type>
NSL::RealTypeOf<Type> getAcceptanceRate(const std::vector<NSL::MCMC::MarkovState<Type>>& MC){
    NSL::RealTypeOf<Type> acceptanceRate = 0;

    for(const auto& state: MC){
        acceptanceRate += static_cast<NSL::RealTypeOf<Type>>( state.accepted );
    }

    return acceptanceRate/MC.size();
}

}

#endif //NSL_MARKOV_STATE_TPP
