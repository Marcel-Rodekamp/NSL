#ifndef NSL_PARAMETER
#define NSL_PARAMETER

#include <sstream>
#include "logger.hpp"
#include "IO/to_string.tpp"
#include "Lattice.hpp"
#include "Lattice/Implementations/square.hpp"
    
#include<iostream>
#include<complex>
#include<variant>
#include<unordered_map>
#include<concepts>
#include<typeinfo>
#include<type_traits>
#include<cxxabi.h> 

namespace NSL{

std::string demangle(const char * tis){
    int status;
    char * buf = abi::__cxa_demangle(tis, nullptr, nullptr, &status);
    std::string out(buf);
    std::free(buf);
    return out;
}

//! Implementation of an Entry object for arbitrary types
/*! 
 * This implementation is basically a std::variant with some syntactic sugar around
 * Usage:
 *
 * ```
 * #include "NSL.hpp"
 * 
 * // declare test function 
 * template<typename Type> f(Type myVal); 
 *
 * int main(){
 *     // initialize a dictionary containing parameters
 *     Parameter p;
 *
 *     // put some parameters
 *     p["p1"] = 2;
 *     p["p2"] = float(2.);
 *
 *     // complex<float> is not an allowed type (see definition of Entry), thus this line throws a compile time error
 *     // p["p3"] = std::complex<float>(2.);
 * 
 *     // printing is possible from the Entry type
 *     std::cout << p["p1"] << std::endl;
 * 
 *     // we can decompose the Entry into a complex, here the int entry is casted to complex<double>
 *     f<std::complex<double>>(p["p1"]);
 * 
 *     return EXIT_SUCCESS;
 * }
 *
 * // This function accepts any argument, requiring that operator<< extists 
 * // If given an Entry object, the corresponding functions from there can be called.
 * // we can further decompose it into a desired type by calling this
 * // `f<DesiredType>(entryArgument)`
 * // where the `entryArgument` is decomposed into DesiredType 
 * template<typename Type>
 * void f(Type myVal){
 *     std::cout << "f<Type=" << demangle(typeid(Type).name()) << ">(myVal=" << myVal << ")" << std::endl;
 * }
 * ```
 * */
template<typename ... Types>
struct EntryImpl{
    EntryImpl() = default;

    //! (Copy-) construct the entry with a given type
    template<typename Type>
    EntryImpl(const Type & entry) : entry(entry) {}

    //! Move construct the entry with a given type
    template<typename Type>
    EntryImpl(Type && entry) : entry(std::move(entry)) {}

    //! Assignment operator allows to copy the contents of one EntryImpl into this
    EntryImpl<Types...> & operator=(const EntryImpl<Types...> & e){
        entry = e.entry;
        return *this;
    }

    //! Assignment operator allows to assign content of type Type to this class
    template<typename Type>
    EntryImpl<Types...> & operator=(const Type & e){
        entry = e;
        return *this;
    }

    //! This operator decomposes the entry into a given type potentially casting it
    template<typename Type>
    operator Type(){
        // we can std::visit to decompose the std::variant into its contained object (C++17)
        // returning a static_cast<Type> version of the contained object allows to 
        // decompose the object into any (castable) type
        // If type cast is not possible (e.g. NSL::Device -> NSL::complex) and tried, a
        // runtime error is thrown. 
        return std::visit(
            [](auto & e){
                if constexpr (std::is_convertible_v<Type,decltype(e)>){
                    return static_cast<Type>(e);
                } else if constexpr (std::is_constructible_v<Type,decltype(e)>){
                    return Type(e);
                } else if constexpr (std::is_same_v<Type, std::string> && std::is_same_v<decltype(e), bool&>) {
                    return Type(e ? "true" : "false");
                } else {
                    throw std::runtime_error(
                        "Entry: Can not convert type(e)="
                        +demangle(typeid(e).name()) 
                        +" to Type="+demangle(typeid(Type).name()) 
                    );
                    return Type();
                }
            },
            entry
        );
    }

    //! This legacy function decomposes the entry into a given type potentially casting it
    template<typename Type>
    Type to(){
        // As we don't want to code the same thing twice, we just call the operator Type() function
        return Type(*this);
    }

    //! Convert the Type of the contained object into a string
    std::string getTypeName() const {
        return std::visit(
            [](auto & e){return demangle(typeid(e).name());},
            entry
        );
    }

    //! Prepare a string representation of the Entry object
    std::string repr() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    //! Prepare a string representation of the stored value
    std::string reprValue() const {
        return std::visit(
            [](auto & e){return NSL::to_string(e);},
            entry
        );
    }

    //! Provide a streaming operator using std::visit
    friend std::ostream & operator<< (std::ostream & os, EntryImpl<Types...> e){
        std::visit(
            [&os,&e](auto & arg){os << "Entry(" << arg << ", type=" << demangle(typeid(arg).name()) << ")";},
            e.entry
        );

        return os;
    }

    // Store the held element in a std::variant
    // The std::variant is the heart of this implementation, basically that is what the EntryImpl boils down to
    std::variant<Types...> entry;
};


//! Put all allowed types for entry
/*!
 * Allowed types are:
 *
 * * Basic Types:
 *      * bool,int, float, double, std::string
 * * NSL Types:
 *      * NSL::Device
 *      * NSL::size_t
 *      * NSL::complex<float> 
 *      * NSL::complex<double>
 *
 * */
typedef EntryImpl<
    // Basic Types 
    bool,
    int,NSL::size_t,
    float,double,
    NSL::complex<float>,NSL::complex<double>,
    std::string,

    // NSL Types
    NSL::Device
> Entry ;

//! A Parameter is a dictionary with trace <string,Entry> where Entry can be any type allowed in the definition Entry
using Parameter = std::unordered_map<std::string, Entry>;

} //namespace NSL
  

//! Provide a formatter for a Entry object based on the implementation of the streaming operator therein.
template <>
struct fmt::formatter<NSL::Entry>: fmt::formatter<std::string> {
    auto format(NSL::Entry e, format_context& ctx) const {
        // convert the Entry into a stringstream
        std::stringstream ss; ss << e;

        // use the standard string formatter to parse the provided string 
        return formatter<std::string>::format(
            fmt::format("{}", ss.str()), ctx
        );
    }
};

#endif // NSL_PARAMETER
