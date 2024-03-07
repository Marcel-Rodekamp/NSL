#ifndef NSL_PARAMETER
#define NSL_PARAMETER

#include <sstream>
#include "logger.hpp"
#include "IO/to_string.tpp"
#include<iostream>
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


// The following code is copied from 
// https://github.com/Marcel-Rodekamp/GeneralType
// which was originally developed for this purpose

//! A function to convert a Type to a string representation.
// ToDo: This requires a lot hard coding, can we automate this?
template<typename Type> 
std::string typeToString(){
    if constexpr( std::is_same_v<Type,bool>){return "bool";}
    
    if constexpr( std::is_same_v<Type,int>){return "int";}

    if constexpr( std::is_same_v<Type,float>){return "float";}
    if constexpr( std::is_same_v<Type,double>){return "double";}
    
    if constexpr( std::is_same_v<Type,NSL::complex<float>>){return "complex<float>";}
    if constexpr( std::is_same_v<Type,NSL::complex<double>>){return "complex<double>";}
    
    if constexpr( std::is_same_v<Type,std::vector<bool>>){return "vector<bool>";}
    if constexpr( std::is_same_v<Type,std::vector<int>>){return "vector<int>";}
    if constexpr( std::is_same_v<Type,std::vector<float>>){return "vector<float>";}
    if constexpr( std::is_same_v<Type,std::vector<double>>){return "vector<double>";}
    if constexpr( std::is_same_v<Type,std::vector<NSL::complex<float>>>){return "vector<complex<float>>";}
    if constexpr( std::is_same_v<Type,std::vector<NSL::complex<double>>>){return "vector<complex<double>>";}
    // ...

    // If type has no explicit implementation we just use the c++ type name
    return demangle(typeid(Type).name());
}


// This anonymous namespace is used to implement the concepts that check weather a type 
// contains certain operators. These are then required to streamline the operators of the 
// GeneralType to the Type specific operators
namespace {

//! A concept that checks weather a type has a shape function `.shape(NSL::size_t)`
template<typename T> 
concept hasShape = requires(T t) {
    { t.shape(0) } -> std::convertible_to<NSL::size_t>;
};

//! A concept that checks weather a type has a streaming operator `operator<<`
template<typename T> 
concept hasStreamingOperator = requires(std::ostream &os, T t) {
    { os << t } -> std::convertible_to<std::ostream &>;
};

//! A concept that checks weather a type has a negation operator `operator!`
template<typename T> 
concept hasNegationOperator = requires(T t) {
    { !t } -> std::convertible_to<bool>;
};

//! A concept that checks weather a type has a negation operator `operator!`
template<typename T> 
concept hasDereferenceOperator = requires(T t) {
    { *t } -> std::convertible_to<decltype(*t)>;
};

//! A concept that checks weather a type has a prefix increment `operator++`
template<typename T> 
concept hasPrefixIncrementOperator = requires(T t) {
    { ++t } -> std::convertible_to<decltype(++t)>;
};

//! A concept that checks weather a type has a postfix increment `operator++(int)`
template<typename T> 
concept hasPostfixIncrementOperator = requires(T t) {
    { t++ } -> std::convertible_to<decltype(t++)>;
};

//! A concept that checks weather a type has a prefix decrement `operator--`
template<typename T> 
concept hasPrefixDecrementOperator = requires(T t) {
    { --t } -> std::convertible_to<decltype(--t)>;
};

//! A concept that checks weather a type has a postfix decrement `operator--(int)`
template<typename T> 
concept hasPostfixDecrementOperator = requires(T t) {
    { t-- } -> std::convertible_to<decltype(t--)>;
};

//! A concept that checks if two types are addable 
template<typename T, typename U>
concept areAddable = requires(T t, U u) {
    { t + u } -> std::convertible_to<decltype(t + u)>;
};

//! A concept that checks if two types are subtractable 
template<typename T, typename U>
concept areSubtractable = requires(T t, U u) {
    { t - u } -> std::convertible_to<decltype(t - u)>;
};

//! A concept that checks if two types are multipliable 
template<typename T, typename U>
concept areMultipliable = requires(T t, U u) {
    { t * u } -> std::convertible_to<decltype(t * u)>;
};

//! A concept that checks if two types are multipliable 
template<typename T, typename U>
concept areDivisible = requires(T t, U u) {
    { t / u } -> std::convertible_to<decltype(t / u)>;
};

//! A concept that checks if two types can be used with the modulus operator 
template<typename T, typename U>
concept areModulus = requires(T t, U u) {
    { t % u } -> std::convertible_to<decltype(t % u)>;
};

//! A concept that checks if two types are bitwise and-able
template<typename T, typename U>
concept areBitwiseAndable = requires(T t, U u) {
    { t & u } -> std::convertible_to<decltype(t & u)>;
};

//! A concept that checks if two types are logical and-able
template<typename T, typename U>
concept areLogicalAndable = requires(T t, U u) {
    { t && u } -> std::convertible_to<decltype(t && u)>;
};

//! A concept that checks if two types are exclusive or-able
template<typename T, typename U>
concept areExclusiveOrable = requires(T t, U u) {
    { t ^ u } -> std::convertible_to<decltype(t ^ u)>;
};

//! A concept that checks if two types are Bitwise inclusive or-able 
template<typename T, typename U>
concept areBitwiseInclusiveOrable = requires(T t, U u) {
    { t | u } -> std::convertible_to<decltype(t | u)>;
};

//! A concept that checks if two types are Logical inclusive or-able 
template<typename T, typename U>
concept areLogicalInclusiveOrable = requires(T t, U u) {
    { t || u } -> std::convertible_to<decltype(t || u)>;
};

//! A concept that checks if two types are smaller-comparable 
template<typename T, typename U>
concept areSmallerComparable = requires(T t, U u) {
    { t < u } -> std::convertible_to<decltype(t < u)>;
};

//! A concept that checks if two types are larger-comparable 
template<typename T, typename U>
concept areLargerComparable = requires(T t, U u) {
    { t > u } -> std::convertible_to<decltype(t > u)>;
};

//! A concept that checks if two types are smaller equal-comparable 
template<typename T, typename U>
concept areSmallerEqualComparable = requires(T t, U u) {
    { t <= u } -> std::convertible_to<decltype(t <= u)>;
};

//! A concept that checks if two types are larger equal-comparable 
template<typename T, typename U>
concept areLargerEqualComparable = requires(T t, U u) {
    { t >= u } -> std::convertible_to<decltype(t >= u)>;
};


//! A concept that checks if two types are equality-comparable 
template<typename T, typename U>
concept areEqualityComparable = requires(T t, U u) {
    { t == u } -> std::convertible_to<decltype(t == u)>;
};

//! A concept that checks if two types are inequality-comparable 
template<typename T, typename U>
concept areInequalityComparable = requires(T t, U u) {
    { t != u } -> std::convertible_to<decltype(t != u)>;
};

//! A concept that checks if two types are add assign-able
template<typename T, typename U>
concept areAddAssignable = requires(T t, U u) {
    { t += u } -> std::convertible_to<decltype(t += u)>;
};

//! A concept that checks if two types are subtract assign-able
template<typename T, typename U>
concept areSubtractAssignable = requires(T t, U u) {
    { t -= u } -> std::convertible_to<decltype(t -= u)>;
};

//! A concept that checks if two types are multiply assign-able
template<typename T, typename U>
concept areMultiplyAssignable = requires(T t, U u) {
    { t *= u } -> std::convertible_to<decltype(t *= u)>;
};

//! A concept that checks if two types are division assign-able
template<typename T, typename U>
concept areDivisionAssignable = requires(T t, U u) {
    { t /= u } -> std::convertible_to<decltype(t /= u)>;
};

//! A concept that checks if two types are modulus assign-able
template<typename T, typename U>
concept areModulusAssignable = requires(T t, U u) {
    { t %= u } -> std::convertible_to<decltype(t %= u)>;
};

//! A concept that checks if two types are bitwise and assign-able
template<typename T, typename U>
concept areBitwiseAndAssignable = requires(T t, U u) {
    { t &= u } -> std::convertible_to<decltype(t &= u)>;
};

//! A concept that checks if two types are bitwise or assign-able
template<typename T, typename U>
concept areBitwiseInclusiveOrAssignable = requires(T t, U u) {
    { t |= u } -> std::convertible_to<decltype(t |= u)>;
};

//! A concept that checks if two types are exclusive or assign-able
template<typename T, typename U>
concept areExclusiveOrAssignable = requires(T t, U u) {
    { t ^= u } -> std::convertible_to<decltype(t ^= u)>;
};

//! A concept that checks if two types are right shift assign-able
template<typename T, typename U>
concept areRightShiftAssignable = requires(T t, U u) {
    { t >>= u } -> std::convertible_to<decltype(t >>= u)>;
};

//! A concept that checks if two types are left shift assign-able
template<typename T, typename U>
concept areLeftShiftAssignable = requires(T t, U u) {
    { t <<= u } -> std::convertible_to<decltype(t <<= u)>;
};

//! A concept that checks if a type has the access operator[]
template<typename T, typename U>
concept areAccessible = requires(T t, U key) {
    { t[key] } -> std::convertible_to<decltype(t[key])>;
};

} // namespace 

/*! 
 * Implementation of a holder class that can hold any of the types specified in the template parameter pack
 * It is a wrapper around the std::variant class that implements a type cast operator to the desired type as well as some convenience functions for simpler usage
 */
template<typename ... Types_>
class GeneralType{
    public:
    //! Default-construct a `GeneralType`
    GeneralType() = default;

    //! Copy-construct a `GeneralType`
    GeneralType( const GeneralType<Types_...> & genT) = default;

    //! Move-construct a `GeneralType`
    GeneralType( GeneralType<Types_...> && genT) = default;

    //! Copy-assign a `GeneralType`
    GeneralType<Types_...> & operator=( const GeneralType<Types_...> & genT) = default;

    //! Move-assign a `GeneralType`
    GeneralType<Types_...> & operator=( GeneralType<Types_...> && genT) = default;

    //! Copy-Construct the GeneralType<Types...> from an object with type Type;
    template<typename Type>
    GeneralType( const Type & obj ) :
        obj_(obj)
    {}

    //! Move-Construct the GeneralType<Types...> from an object with type Type;
    template<typename Type>
    GeneralType( Type && obj ) :
        obj_(std::move(obj))
    {}

    //! Copy-assign the GeneralType<Types...> from an object with type Type;
    template<typename Type>
    GeneralType<Types_...> & operator=( const Type & obj ){
        obj_ = obj;
        return *this;
    }

    //! Move-assign the GeneralType<Types...> from an object with type Type;
    template<typename Type>
    GeneralType<Types_...> & operator=( Type && obj ){
        obj_ = obj;
        return *this;
    }

    //! This operator decomposes the `GeneralType` into a given type potentially casting it
    template<typename Type>
    operator Type(){
        return std::visit(
            [](auto & e){
                if constexpr (std::is_convertible_v<Type,decltype(e)>){
                    return static_cast<Type>(e);
                } else if constexpr (std::is_constructible_v<Type,decltype(e)>){
                    return Type(e);
                } else {
                    throw std::runtime_error(
                        "Can not convert held type ("
                        + typeToString<decltype(e)>()
                        +") to desired Type ("+typeToString<Type>()+")"
                    );
                    return Type();
                }
            },
            obj_
        );
    }

    //! This operator decomposes the `GeneralType` into a given type potentially casting it
    template<typename Type>
    Type to(){return *this;}

    //! An implementation that puts the content of `GeneralType` to the out stream `os`
    friend std::ostream & operator<< (std::ostream & os, GeneralType<Types_...> genT){
        std::visit(
            [&os,&genT](auto & arg){
                if constexpr (hasStreamingOperator<decltype(arg)>){
                    os << arg;
                } else {
                    throw std::runtime_error(
                        "Can not  invoke operator<< held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                }
            },
            genT.obj_
        );

        return os;
    }

    // The following part of this class implements different operators that can come in handy. The general idea is
    // if the contained type implements the operator then the General Type should call that otherwise evaluate a static_assert

    // =========================================================================================
    // Unary Operators
    // =========================================================================================
    
    //! Negation operator, forwards to the negation operator of the held type
    GeneralType<Types_...> operator!(){
        return std::visit(
            [](const auto & arg){
                if constexpr( hasNegationOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(!arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator! on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Dereference operator, forwards to the dereference operator of the held type
    GeneralType<Types_...> operator*(){
        return std::visit(
            [](const auto & arg){
                if constexpr( hasDereferenceOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(*arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator! on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Prefix increment operator, forwards to the prefix increment operator of the held type
    GeneralType<Types_...> operator++(){
        return std::visit(
            [](auto & arg){
                if constexpr( hasPrefixIncrementOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(++arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke prefix operator++ on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Postfix increment operator, forwards to the prefix increment operator of the held type
    GeneralType<Types_...> operator++(int){
        return std::visit(
            [](auto & arg){
                if constexpr( hasPostfixIncrementOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(arg++);
                } else {
                    throw std::runtime_error(
                        "Can not invoke postfix operator++ on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Prefix decrement operator, forwards to the prefix increment operator of the held type
    GeneralType<Types_...> operator--(){
        return std::visit(
            [](auto & arg){
                if constexpr( hasPrefixIncrementOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(--arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke prefix operator++ on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Postfix decrement operator, forwards to the prefix increment operator of the held type
    GeneralType<Types_...> operator--(int){
        return std::visit(
            [](auto & arg){
                if constexpr( hasPostfixIncrementOperator<decltype(arg)> ){
                    return GeneralType<Types_...>(arg--);
                } else {
                    throw std::runtime_error(
                        "Can not invoke postfix operator-- on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! Postfix decrement operator, forwards to the prefix increment operator of the held type
    GeneralType<Types_...> shape(NSL::size_t dim){
        return std::visit(
            [&dim](auto & arg){
                if constexpr( hasShape<decltype(arg)> ){
                    return GeneralType<Types_...>(arg.shape(dim));
                } else {
                    throw std::runtime_error(
                        "Can not invoke .shape(dim) on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }


    //! Access operator, forwards to the access operator of the held type
    GeneralType<Types_...> operator[](const auto & key){
        return std::visit(
            [&key](auto & arg){
                if constexpr( areAccessible<decltype(arg),decltype(key)> ){
                    return GeneralType<Types_...>(arg[key]);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator[] on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }

    //! const Access operator, forwards to the access operator of the held type
    GeneralType<Types_...> operator[](const auto & key) const {
        return std::visit(
            [&key](auto & arg){
                if constexpr( areAccessible<decltype(arg),decltype(key)> ){
                    return GeneralType<Types_...>(arg[key]);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator[] on held type (" 
                        + typeToString<decltype(arg)>() + ")"
                    );
                    return GeneralType<Types_...>(arg);
                }
            },
            this->obj_
        );
    }


    // =========================================================================================
    // Binary Operators
    // =========================================================================================

    
    //! Addition operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator+(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areAddable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg + rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator+ on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Subtraction operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator-(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areSubtractable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            // This operation somehow requires long int in the std::variant
                            // I don't understand why, but I added it as a fix as default 
                            // to the obj_
                            return GeneralType<Types_...>( 
                                lhs_arg - rhs_arg
                            );
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator- on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Multiplication operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator*(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areMultipliable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg * rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator* on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Division operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator/(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areDivisible<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg / rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator/ on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Modulus operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator%(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areModulus<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg % rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator% on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Bitwise AND operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator&(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areBitwiseAndable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg & rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator& on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Logical AND operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator&&(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areLogicalAndable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg && rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator&& on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Exclusive Or operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator^(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areExclusiveOrable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg ^ rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator^ on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Bitwise inclusive Or operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator|(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areExclusiveOrable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg | rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator| on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Logical inclusive Or operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator||(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areLogicalInclusiveOrable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg || rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator|| on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison smaller operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator<(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areSmallerComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg < rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator< on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison larger operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator>(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areLargerComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg > rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator> on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison smaller equal operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator<=(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areSmallerEqualComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg <= rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator<= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison larger equal operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator>=(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areLargerEqualComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg >= rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator>= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison Equality operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator==(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areEqualityComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg == rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator== on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Comparison Inequality operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator!=(GeneralType<Types_...> rhs){
        return std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areInequalityComparable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            return GeneralType<Types_...>(lhs_arg != rhs_arg);
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator=! on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                            return GeneralType<Types_...>(lhs_arg);
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );
    }

    //! Addition assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator+=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areAddAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg += rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator+= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }
 
    //! Subtraction assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator-=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areSubtractAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg -= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator-= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Multiplication assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator*=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areMultiplyAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg *= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator*= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Division assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator/=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areDivisionAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg /= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator/= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Modulus assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator%=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr ( areModulusAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg %= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator%= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Bitwise AND assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator&=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg] (auto & rhs_arg){
                        if constexpr (areBitwiseAndAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg &= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator&= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Bitwise Inclusive OR assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator|=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs,this](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg,this] (auto & rhs_arg){
                        if constexpr (areBitwiseInclusiveOrAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg |= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator|= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    //! Exclusive OR assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator^=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs,this](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg,this] (auto & rhs_arg){
                        if constexpr (areExclusiveOrAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg ^= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator^= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }
    
    //! Right shift assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator>>=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs,this](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg,this] (auto & rhs_arg){
                        if constexpr (areRightShiftAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg >>= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator>>= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }   
    
    //! Left shift assignment operator, forwards to the addition operator of the held type
    GeneralType<Types_...> operator<<=(GeneralType<Types_...> rhs){
        std::visit(
            [&rhs,this](auto & lhs_arg){
                return std::visit(
                    [&lhs_arg,this] (auto & rhs_arg){
                        if constexpr (areLeftShiftAssignable<decltype(lhs_arg),decltype(rhs_arg)> ){
                            lhs_arg <<= rhs_arg;
                        } else {
                            throw std::runtime_error(
                                "Can not invoke operator<<= on held types (" 
                                + typeToString<decltype(lhs_arg)>() + " and " 
                                + typeToString<decltype(rhs_arg)>() + ")"
                            );
                        }
                    },
                    rhs.obj_
                );
            },
            this->obj_
        );

        return *this;
    }

    // =========================================================================================
    // External Operators
    // =========================================================================================
    
    //! A function, that checks if `Type` is held by provided GeneralType
    template<typename Type, typename ... Types>
    friend constexpr bool holdsType( const GeneralType<Types...> & gt ){
        if constexpr( (std::is_same_v<Type,Types> || ... ) ){
            return std::holds_alternative<Type>(gt.obj_);
        } else {
            return false;
        }
    }

    //! Addition operator with non-GeneralType, forwards to the addition operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator+(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areAddable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS + rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator+(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );

                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Subtraction operator with non-GeneralType, forwards to the subtraction operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator-(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areSubtractable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS - rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator-(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Multiplication operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator*(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areMultipliable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS * rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator*(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Devision operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator/(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areDivisible<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS / rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator/(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Modulus operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator%(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areModulus<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS % rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator%(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Bitwise AND operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator&(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areBitwiseAndable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS & rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator&(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Logically AND operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator&&(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areLogicalAndable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS && rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator&&(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Bitwise inclusive OR operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator|(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areBitwiseInclusiveOrable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS | rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator|(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Exclusive OR operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator^(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areExclusiveOrable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS ^ rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator^(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Logical inclusive OR operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator||(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areLogicalInclusiveOrable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS || rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator||(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Smaller operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator<(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areSmallerComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS < rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator<(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Larger operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator>(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areLargerComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS > rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator>(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Smaller-Equal operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator<=(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areSmallerEqualComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS <= rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator<=(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Larger-Equal operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend GeneralType<Types...> operator>=(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areLargerEqualComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return GeneralType<Types...>(LHS >= rhs_arg);
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator>=(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return GeneralType<Types...>(rhs_arg);
                }    
            },
            RHS.obj_
        );
    }

    //! Equality operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend bool operator==(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areEqualityComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return LHS == rhs_arg;
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator==(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return false;
                }    
            },
            RHS.obj_
        );
    }


    //! Equality operator with non-GeneralType, forwards to the multiplication operator of the held type
    template<typename Type, typename ... Types>
        requires((std::is_same_v<Type,Types> || ... ))
    friend bool operator!=(const Type & LHS, const GeneralType<Types...> & RHS ){
        return std::visit(
            [&LHS](auto & rhs_arg){
                if constexpr (areInequalityComparable<decltype(LHS),decltype(rhs_arg)> ){
                    return LHS != rhs_arg;
                } else {
                    throw std::runtime_error(
                        "Can not invoke operator!=(Type, GeneralType) on held types (" 
                        + typeToString<decltype(LHS)>() + " and " 
                        + typeToString<decltype(rhs_arg)>() + ")"
                    );
    
                    return false;
                }    
            },
            RHS.obj_
        );
    }

    protected:

    // Store the held element in a std::variant
    // The std::variant is the heart of this implementation, basically that is what the EntryImpl boils down to
    // The long int implements a fix for the subtraction operator. I don't understand why it is needed, but it works
    std::variant<Types_...> obj_;
}; // GeneralType<Types_...>

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
typedef GeneralType<
    // Basic Types 
    bool,
    int,NSL::size_t,
    float,double,
    NSL::complex<float>,NSL::complex<double>,
    std::string,
    NSL::Tensor<double>,
    NSL::Tensor<NSL::complex<double>>,

    // NSL Types
    NSL::Device
>  GenType;

//! A Parameter is a dictionary with trace <string,Entry> where Entry can be any type allowed in the definition Entry
using Parameter = std::unordered_map<
    std::string, 
    GenType
>;

} //namespace NSL
  

//! Provide a formatter for a Entry object based on the implementation of the streaming operator therein.
template <>
struct fmt::formatter<NSL::GenType>: fmt::formatter<std::string> {
    auto format(NSL::GenType e, format_context& ctx) const {
        // convert the Parameter into a stringstream
        std::stringstream ss; ss << e;

        // use the standard string formatter to parse the provided string 
        return formatter<std::string>::format(
            fmt::format("{}", ss.str()), ctx
        );
    }
};

#endif // NSL_PARAMETER
