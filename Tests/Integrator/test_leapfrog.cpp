#include "../test.hpp"
#include "IO/to_string.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "complex.hpp"
#include "types.hpp"

//! Test that the leapfrog is reversible
template<NSL::Concept::isNumber Type, class Action>
void reversibility(Action & action, std::pair<NSL::size_t,NSL::size_t> fieldShape, NSL::size_t trajectoryLength, NSL::size_t numberSteps);

//! Test that the leapfrog is energy preserving
template<NSL::Concept::isNumber Type, class Action>
void energyPreservation(Action & action, std::pair<NSL::size_t,NSL::size_t> fieldShape, NSL::size_t trajectoryLength, NSL::size_t numberSteps);

FLOAT_NSL_TEST_CASE("Leapfrog Reversibility", "[Integrator,Leapfrog]"){
    for(NSL::size_t Nt = 1; Nt < 12; Nt*=2){
        NSL::Parameter params;
        params["beta"] = 1.;
        params["U"] = 1.;
        params["Nt"] = Nt;

	    NSL::Action::Action S = NSL::Action::HubbardGaugeAction<TestType>(params);

        for(NSL::size_t Nx = 1; Nx < 12; Nx*=2){
            for(NSL::size_t trajectoryLength = 1; trajectoryLength <= 3; trajectoryLength+=1){
                for(NSL::size_t Nmd = 1; Nmd <= 16; Nmd*=2){
                    reversibility<TestType>(S, {Nt,Nx}, trajectoryLength, Nmd);
                }
            }
        }
    }
}

FLOAT_NSL_TEST_CASE("Leapfrog Energy Preservation", "[Integrator,Leapfrog]"){
    for(NSL::size_t Nt = 1; Nt < 12; Nt*=2){
        NSL::Parameter params;
        params["beta"] = 1.;
        params["U"] = 1.;
        params["Nt"] = Nt;

	    NSL::Action::Action S = NSL::Action::HubbardGaugeAction<TestType>(params);

        for(NSL::size_t Nx = 1; Nx < 12; Nx*=2){
            for(float trajectoryLength = 0.1; trajectoryLength <= 2.; trajectoryLength+=0.5){
                for(NSL::size_t Nmd = 1; Nmd <= 32; Nmd*=2){
                    if (0.5 <= trajectoryLength/Nmd){
                        continue;
                    }
                    energyPreservation<TestType>(S, {Nt,Nx}, trajectoryLength, Nmd);
                }
            }
        }
    }
}

/************************************************************************
Reversibility implementation
************************************************************************/

template<NSL::Concept::isNumber Type, class Action>
void reversibility(Action & action, std::pair<NSL::size_t,NSL::size_t> fieldShape, NSL::size_t trajectoryLength, NSL::size_t numberSteps){
    NSL::Tensor<Type> phi(fieldShape.first,fieldShape.second); phi.rand(); 
    NSL::Tensor<Type> pi(fieldShape.first,fieldShape.second); pi.rand(); 

    // define configuration
	NSL::Configuration<Type> config{{"phi",phi}};

    // define momentum
    NSL::Configuration<Type> moment{{"phi",pi}};

    //define forward (F) and backward (B) leapfrog integration
    NSL::Integrator::Leapfrog F_integrator(action,trajectoryLength,numberSteps,false);
    NSL::Integrator::Leapfrog B_integrator(action,trajectoryLength,numberSteps,true );

    // perform forward integration
    auto [pconfig,pmoment] = F_integrator(config,moment);
    // perform backward integration
    auto [bconfig,bmoment] = B_integrator(pconfig,pmoment);

    // compare backward result with initial input
    INFO( std::string("T = ") + std::to_string(trajectoryLength) );
    INFO( std::string("Nmd = ") + std::to_string(numberSteps) );
    INFO( std::string("shape = (") + std::to_string(fieldShape.first)  + std::string(", ") + std::to_string( fieldShape.second ) + std::string(")")  );
    INFO( config["phi"] );
    INFO( bconfig["phi"] );
    INFO( config["phi"] - bconfig["phi"] );
    REQUIRE(almost_equal(bconfig["phi"],config["phi"],std::numeric_limits<Type>::digits10-3).all());
    REQUIRE(almost_equal(bmoment["phi"],moment["phi"],std::numeric_limits<Type>::digits10-3).all());
}

template<NSL::Concept::isNumber Type, class Action>
void energyPreservation(Action & action, std::pair<NSL::size_t,NSL::size_t> fieldShape, NSL::size_t trajectoryLength, NSL::size_t numberSteps){
    NSL::Tensor<Type> phi(fieldShape.first,fieldShape.second); phi.rand(); 
    NSL::Tensor<Type> pi(fieldShape.first,fieldShape.second); pi.rand(); 

    // define configuration
	NSL::Configuration<Type> config{{"phi",phi}};

    // define momentum
    NSL::Configuration<Type> moment{{"phi",pi}};

    //define forward (F) and backward (B) leapfrog integration
    NSL::Integrator::Leapfrog integrator(action,trajectoryLength,numberSteps);

    // perform forward integration
    auto [pconfig,pmoment] = integrator(config,moment);

    // compare backward result with initial input
    INFO( std::string("T = ") + std::to_string(trajectoryLength) );
    INFO( std::string("Nmd = ") + std::to_string(numberSteps) );
    INFO( std::string("shape = (") + std::to_string(fieldShape.first)  + std::string(", ") + std::to_string( fieldShape.second ) + std::string(")") );
    INFO( config["phi"] );
    INFO( pconfig["phi"] );

    Type H_old = static_cast<Type>(0.5)*( moment["phi"] *  moment["phi"]).sum() + action(config); 
    Type H_new = static_cast<Type>(0.5)*(pmoment["phi"] * pmoment["phi"]).sum() + action(pconfig);

    NSL::RealTypeOf<Type> eps = static_cast<NSL::RealTypeOf<Type>>(trajectoryLength) / static_cast<NSL::RealTypeOf<Type>>(numberSteps);
    NSL::RealTypeOf<Type> err = std::abs( (H_old-H_new)/H_old );

    NSL::size_t order_eps = std::floor( std::log10(eps*eps) );
    NSL::size_t order_err = std::floor( std::log10(err) );

    INFO( std::string("Error = ") + NSL::to_string(err) + std::string(" -> ") + NSL::to_string(order_err) );
    INFO( std::string("Tolerance = ") + NSL::to_string(eps*eps) + std::string(" -> ") + NSL::to_string(order_eps) );
    INFO( "It might fail by chance, please check if Error ≈ Tolerance & repeat" )
    REQUIRE(order_err <= order_eps);
}

