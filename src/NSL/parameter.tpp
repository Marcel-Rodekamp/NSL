#ifndef NSL_PARAMETER
#define NSL_PARAMETER

#include <sstream>

    
namespace NSL{

template<typename Type> class Parameter;

// The following structure is inspired by
// https://github.com/Marcel-Rodekamp/ParameterFileManager

class ParameterBase{
    public:
    virtual std::string to_string() = 0;
    virtual void fromString(const std::string &) = 0;
    
    template<typename Type>
    Type to(){
        Parameter<Type> * ptr = dynamic_cast<Parameter<Type> *>(this);
        if(ptr){
            return ptr->template to_<Type>();
        } else {
            throw std::runtime_error("Converting ParameterBase to Parameter failed in ParameterBase::to<Type>() ...");
        }
    }


    friend std::ostream& operator<<(std::ostream & os, ParameterBase * pb){
        os << pb->to_string();
        return os;
    }

};

template<typename Type>
class Parameter: public ParameterBase {
    public:
    Parameter() = default;
    Parameter(const Parameter<Type> &) = default;
    Parameter(Parameter<Type> &&) = default;

    Parameter(const Type & value):
        value_(value)
    {}

    void fromString(const std::string & strVal) override {
        std::stringstream ss(strVal);
        ss >> value_;
    }

    std::string to_string() override {
        return NSL::to_string(this->value_);
    }

    friend std::ostream & operator<<(std::ostream & os, const Parameter<Type> & param){
        os << param.value_; 
        return os;
    }

    friend ParameterBase;

    protected:

    template<typename Type_>
    Type_ to_(){
        return value_;
    }

    Type value_;
};

class ParameterList: public std::unordered_map<std::string, ParameterBase *> {
    public:

    friend std::ostream & operator<<(std::ostream & os, const ParameterList & params){
        for(const auto & [key,param]: params){
            os << key << ": " << param << '\n';
        }
        return os;
    }
};


/*
    // Inspired by https://gieseanw.wordpress.com/2017/05/03/a-true-heterogeneous-container-in-c/
class ErasedContainer{
    public:
    ErasedContainer() = default;

    template<typename Type>
    ErasedContainer(const Type & value)
    {
        containerMem_<Type>[this] = value;        
    }

    template<typename Type>
    operator Type(){
        return containerMem_<Type>[this];
    }

    template<typename Type>
    Type to(){
        return containerMem_<Type>[this];
    }

    private:
    template<typename Type>
    static std::unordered_map<ErasedContainer *, Type> containerMem_;

};

template<typename Type>
std::unordered_map<ErasedContainer *, Type> ErasedContainer::containerMem_;



//! dictionary <string,any> storing any set of id,parameter value pairs
class Parameter: public std::unordered_map<std::string,ErasedContainer>{};

*/
} //namespace NSL

#endif // NSL_PARAMETER
