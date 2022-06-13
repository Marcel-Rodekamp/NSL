#ifndef NSL_CONFIGURATION_TPP
#define NSL_CONFIGURATION_TPP

//! \file configuration.tpp

#include "../Tensor.hpp"
#include "../concepts.hpp"
#include "../map.hpp"

//! \todo: std::any is a comparable slow object. If we want/need to optimize
//         this class considere using std::variant instead.
#include <any>
#include <unordered_map>
#include <algorithm>

namespace NSL {
/*! Configuration storing a key(Field Name)-value(Field) pair
 *
 *  The keys are stored as std::strings while the values can be any type
 *  determined by TensorTypes, we restrict to the use of NSL::Tensor to 
 *  represent any field used in the MCMC implementation. 
 * */
template<typename ... TensorTypes>
class Configuration {
    public:

    //! Helper to enable non-templated access
    /*!
     *  The return type deduction of `Configuration::field` requires
     *  that either `TensorTypes` are all the same or there is only one TensorType specified.
     *  In these cases `Configuration::isHomogeneous::value` is true and
     *  is used to enable `field[]`. Otherwise, `Configuration::isHomogeneous::value`
     *  is false and the `field<TensorType>[]` must be used.
     *
     *  `isHomogeneous::type` is aliasing the first template argument `TensorType`
     * */
    template<typename TensorType, typename ... OtherTensorTypes>
    struct isHomogeneous {
        static constexpr bool value = (std::is_same<TensorType,OtherTensorTypes>::value && ...) || sizeof...(TensorTypes) == 1;
        using type = NSL::Tensor<TensorType>;       
    };
    
 
    //! Configuration creation with key-value pair.
    /*! 
     * The Configuration is defined as a set of fields (NSL::Tensor) 
     *
     * */
    Configuration(std::pair<std::string,NSL::Tensor<TensorTypes>> ... keyVal) :
        dict_{ {std::get<0>(keyVal),std::get<1>(keyVal)} ... }
    {}

	Configuration zero(){
		for(auto fName: fieldNames()){
			field(fName) = zeros_like(field(fName));
		}
		return *this;
	}
	//! Access the field with a given key
    /*!
     * This function requires to specify the desired dtype which should 
     * match the original dtype of the field.
     * There is no (runtime) deduction of the return type.
     * */
    template<typename TensorType>
    NSL::Tensor<TensorType> field(std::string key){
        assert( dict_[key].type() == typeid(NSL::Tensor<TensorType>) );
        return std::any_cast<NSL::Tensor<TensorType>>(dict_[key]);
    }
    
	// Configuration<TensorTypes ...> & operator=(const Configuration<TensorTypes ...> & other){
	// 	int i = 0;
	// 	for(std::string fName : fieldNames()) {
	// 		dict_[fName] = NSL::Tensor(other.field<NthTypeOf<i, TensorTypes>>(fName), true);
	// 		i++;
	// 	}
	// 	return *this;
    // }
    //! Acces the field with a given key (Homogeneous Configuration only)
    /*!
     * Provided as convenience function. If the configuration is considered
     * Homogeneous (all fields have the same type, see `Configuration::isHomogeneous`)
     * a non templated access method is available. 
     * */
    typename isHomogeneous<TensorTypes...>::type field(std::string key)
        requires ( isHomogeneous<TensorTypes...>::value )
    {
        return std::any_cast<typename isHomogeneous<TensorTypes...>::type>(dict_[key]);
    }

    //! Access the field names
    std::array<std::string,sizeof...(TensorTypes)> fieldNames(){
        std::array<std::string,sizeof...(TensorTypes)> names;
    
        std::transform(
            this->dict_.begin(),
            this->dict_.end(),
            names.begin(),
            [](const auto & x) {return x.first;}
        );
        
        return std::move(names);
    }

    private:
    std::unordered_map<std::string, std::any> dict_;

};
} // namespace NSL

#endif // NSL_CONFIGURATION_TPP
