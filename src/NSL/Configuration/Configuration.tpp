#ifndef NSL_CONFIGURATION_TPP
#define NSL_CONFIGURATION_TPP

#include "../concepts.hpp"
#include "../Tensor.hpp"

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

    private:
};

} // namespace NSL

#endif //NSL_CONFIGURATION_TPP
