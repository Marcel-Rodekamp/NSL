#ifndef NSL_RADIAL_MARKOV_STATE_TPP
#define NSL_RADIAL_MARKOV_STATE_TPP

#include <concepts>
#include <type_traits>
#include "Configuration/Configuration.tpp"
#include "complex.hpp"
#include "types.hpp"

namespace NSL::MCMC {


template<NSL::Concept::isNumber Type>
class RadialMarkovState{
    public:

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    RadialMarkovState(
        NSL::Configuration<Type> config,
        const Type & actionValue,
        const NSL::RealTypeOf<Type> & acceptanceProbability,
        NSL::size_t markovTime,
        bool accepted,
        bool radialStep,
        bool radiusIncreased,
        WeightType ... weights
    ): 
        configuration(config),
        actionValue(actionValue),
        weights({weights...}),
        acceptanceProbability(acceptanceProbability),
        markovTime(markovTime),
        accepted(accepted),
        radialStep(radialStep),
        radiusIncreased(radiusIncreased) 
    {}

    template<std::convertible_to<std::pair<std::string, Type>> ... WeightType>
    RadialMarkovState(
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
        accepted(1),
        radialStep(0),
        radiusIncreased(0)

    {}

    RadialMarkovState() = default;

    RadialMarkovState( const RadialMarkovState<Type> & ) = default;
    RadialMarkovState( RadialMarkovState<Type> && ) = default;

    RadialMarkovState<Type>& operator=(const RadialMarkovState<Type>&) = default;
    RadialMarkovState<Type>& operator=(RadialMarkovState<Type>&&) = default;

    //! Store the configuration associated with the Markov State
    NSL::Configuration<Type> configuration;

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
    bool radialStep;
    bool radiusIncreased; // If the update is a radial step record whether it was proposed to increase the radius
};

template<NSL::Concept::isNumber Type>
NSL::RealTypeOf<Type> getAcceptanceRate(const std::vector<NSL::MCMC::RadialMarkovState<Type>>& MC){
    NSL::RealTypeOf<Type> acceptanceRate = 0;
    NSL::RealTypeOf<Type> HMCsteps = 0;

    for(const auto& state: MC){
        if (! state.radialStep){
            HMCsteps += 1;
            acceptanceRate += static_cast<NSL::RealTypeOf<Type>>( state.accepted );
        }
    }

    return acceptanceRate/HMCsteps;
}

}

#endif //NSL_RADIAL_MARKOV_STATE_TPP
