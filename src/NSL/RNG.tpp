#ifndef NSL_RNG_STUFF_SOMETHING_FUNNY_SOMEBODY_HELP_MEEEEEEE_TPP
#define NSL_RNG_STUFF_SOMETHING_FUNNY_SOMEBODY_HELP_MEEEEEEE_TPP

#include <random>

namespace NSL {
    namespace{
	std::mt19937_64 RNG_GENERATOR; // sick. ask marcel.
    };

    
    template <NSL::Concept::isNumber Type>
    class Random {
	
    public:
	Random(std::size_t seed64){
	    std::cout<<"initializing"<<std::endl;
	    if (initialize()){
		RNG_GENERATOR.seed(seed64);
		initialize()--;
		std::cout<<"initialized"<<std::endl;
	    } 
	}

	Random() : Random(std::random_device()()) {}

	Type uni_dis_rng(){
	    static std::uniform_real_distribution<NSL::RealTypeOf<Type>>uni_dis(0.0,1.0);
	    static std::normal_distribution<NSL::RealTypeOf<Type>>nml_dis(0.0,1.0);
	    if constexpr(NSL::Concept::isFloatingPoint<Type>){
		if constexpr(NSL::Concept::isComplex<Type>){
		    return NSL::complex(uni_dis(RNG_GENERATOR), uni_dis(RNG_GENERATOR));
		} else {
		    return uni_dis(RNG_GENERATOR);
		}
	    } else {
		throw;
	    }
	}
	
	Type nml_dis_rng(){
	    static std::uniform_real_distribution<NSL::RealTypeOf<Type>>uni_dis(0.0,1.0);
	    static std::normal_distribution<NSL::RealTypeOf<Type>>nml_dis(0.0,1.0);
	    if constexpr(NSL::Concept::isFloatingPoint<Type>){
		if constexpr(NSL::Concept::isComplex<Type>){
		    return NSL::complex(nml_dis(RNG_GENERATOR), nml_dis(RNG_GENERATOR));
		} else {
		    return nml_dis(RNG_GENERATOR);
		}
	    } else {
		throw;
	    }
	}
	    
	Type uni_dis_lo_hi_rng(NSL::size_t low, NSL::size_t high){
	    if constexpr (NSL::Concept::isIntegral<Type>){
		static NSL::size_t lo=0;
		static NSL::size_t hi=10;
		static std::uniform_int_distribution<Type> dis(lo,hi);
		
		if ( ( high == hi ) && ( low == lo ) ){
		    return dis(RNG_GENERATOR);
		} else {
		    hi = high;
		    lo = low;
		    dis = std::uniform_int_distribution<Type>(lo, hi);
		    return dis(RNG_GENERATOR);
		    
		}
		throw;
	    } else { //isIntegral
		throw;
	    }
	} 

	Type uni_dis_hi_rng(NSL::size_t high){
	    if constexpr (NSL::Concept::isIntegral<Type>){
		static NSL::size_t hi=10;
		static std::uniform_int_distribution<Type> dis(0,hi);
		
		if ( high == hi ) {
		    return dis(RNG_GENERATOR);
		} else {
		    hi = high;
		    dis = std::uniform_int_distribution<Type>(0, hi);
		    return dis(RNG_GENERATOR);
		    
		}
		throw;
	    } else { //isIntegral
		throw;
	    }
	} 
	    
	
	private:
	static auto initialize() -> std::size_t& {
	    static std::size_t init = 1;
	    return init;
	}

    };
}
#endif
