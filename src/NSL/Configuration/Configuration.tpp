#ifndef NSL_CONFIGURATION_TPP
#define NSL_CONFIGURATION_TPP

#include "../concepts.hpp"
#include "../Tensor.hpp"
#include "typePromotion.hpp"

#include<unordered_map>
#include<string>

namespace NSL {

//! Configuration
/*!
 *  The configuration class is a container, special type of a dictionary, 
 *  providing a key,value pair for particular fields.
 *
 *  It is fully STL comparible.
 * */
template<NSL::Concept::isNumber Type>
class Configuration : public std::unordered_map<std::string, NSL::Tensor<Type>> {
    public:
        using std::unordered_map<std::string,NSL::Tensor<Type>>::unordered_map;

        //! deepcopy constructor
        Configuration( const Configuration<Type> & other, bool deep_copy ) : std::unordered_map<std::string, NSL::Tensor<Type>>() {
            for(auto & [key,field]: other){
                NSL::Tensor<Type> tmp(field,deep_copy);
                this->operator[](key) = std::move(tmp);
            }
        }

        //! Add a configuration in place
        /*! 
         * If this contains field from other: add
         * If this doesn't contain field from other: append
         * */
        Configuration<Type> & operator += ( const Configuration<Type> & other ){
            for(auto &[key,field]: other){
                if(this->contains(key)){
                    this->operator[](key) += field;
                } else {
                    this->operator[](key) = field;
                }
            } 

            return *this;
        }
        template<NSL::Concept::isNumber OtherType>
        auto & operator += ( const Configuration<OtherType> & other ){
            for(auto &[key,field]: other){
                if(this->contains(key)){
                    this->operator[](key) += field;
                } else {
                    this->operator[](key) = field;
                }
            } 

            return *this;
        }

        //! Subtract a configuration in place
        /*! 
         * If this contains field from other: add
         * If this doesn't contain field from other: append
         * */
        Configuration<Type> & operator -= ( const Configuration<Type> & other ){
            for(auto &[key,field]: other){
                if(this->contains(key)){
                    this->operator[](key) -= field;
                } else {
                    this->operator[](key) = -field;
                }
            } 

            return *this;
        }
        template<NSL::Concept::isNumber OtherType>
        auto & operator -= ( const Configuration<OtherType> & other ){
            for(auto &[key,field]: other){
                if(this->contains(key)){
                    this->operator[](key) -= field;
                } else {
                    this->operator[](key) =-field;
                }
            } 

            return *this;
        }



        //! Streaming operator
        friend std::ostream & operator<<(std::ostream & os, const Configuration<Type> & conf){
            for(const auto &[key,field]: conf){
                os << "Configuration(" << key << "): \n" << field << "\n";
            }
            return os;
        }
    private:
};

//! Add a configuration in place
/*! 
* If this contains field from other: add
* If this doesn't contain field from other: append
* */
template<NSL::Concept::isNumber Type>
Configuration<Type> operator+( const Configuration<Type> & lhs, 
                               const Configuration<Type> & rhs )
{
    Configuration<Type> tmp(lhs,true);
    tmp+=rhs;
    return std::move(tmp);
}

//! Subtract a configuration in place
/*! 
* If this contains field from other: add
* If this doesn't contain field from other: append
* */
template<NSL::Concept::isNumber Type>
Configuration<Type> operator-( const Configuration<Type> & lhs, 
                               const Configuration<Type> & rhs )
{
    Configuration<Type> tmp(lhs,true);
    tmp-=rhs;
    return std::move(tmp);
}

//! Multiply a configuration by a number 
/*!
 * Multiply each field by a number
 * */
template<NSL::Concept::isNumber NumberType, NSL::Concept::isNumber ConfigType>
auto operator*( const Configuration<ConfigType> config, const NumberType & number) {
    Configuration<ConfigType> config_(config,true);
    for(auto & [key,field] : config_ ){
        field *= number;
    }
    return config_;
}

//! Multiply a configuration by a number 
/*!
 * Multiply each field by a number
 * */
template<NSL::Concept::isNumber NumberType, NSL::Concept::isNumber ConfigType>
auto operator*(const NumberType & number, const Configuration<ConfigType> config) {
    Configuration<ConfigType> config_(config,true);
    for(auto & [key,field] : config_ ){
        field *= number;
    }
    return config_;
}

} // namespace NSL

#endif //NSL_CONFIGURATION_TPP
