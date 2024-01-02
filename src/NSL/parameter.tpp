#ifndef NSL_PARAMETER
#define NSL_PARAMETER

#include <sstream>
#include "IO/to_string.tpp"
    
namespace NSL{

template<typename Type> class TemplatedParameterEntry;

/*!
 * This class represents a single entry in `NSL::Parameter(std::unordered_map<std::string,ParameterEntry *>)`.
 * It implements basic functionality to hide the data type of the parameter stored.
 * This is achieved by Deriving from this class (see `NSL::TemplatedParameterEntry`) and 
 * dynamic casting to the child class.
 * This Base class is the point of access. The user should never have to
 * handle `NSL::TemplatedParameterEntry`.
 *
 * \todo: Implement a way to do `ParameterEntry(myVariable)`;
 *        such that it can be used together with `NSL::Parameter::operator[]`;
 * */
class ParameterEntry{
    public:
    ParameterEntry() = default;

    //! Retrieve the stored value. Type must match the initial type otherwise runtime error is thrown.
    template<typename Type>
    Type & to(){
        // get the derived TemplatedParameterEntry
        TemplatedParameterEntry<Type> * ptr = upcast<Type>();
        return ptr->value_;
    }

    //! Retrieve the stored value. Type must match the initial type otherwise runtime error is thrown.
    template<typename Type>
    const Type & to() const {
        // get the derived TemplatedParameterEntry
        TemplatedParameterEntry<Type> * ptr = upcast<Type>();
        return ptr->value_;
    }

    //! Explicitly convert to Type. Type must match the initial type otherwise runtime error is thrown.
    template<typename Type>
    operator Type(){
        // get the derived TemplatedParameterEntry
        TemplatedParameterEntry<Type> * ptr = upcast<Type>();
        return ptr->value_;
    }

    //! Explicitly convert to Type. Type must match the initial type otherwise runtime error is thrown.
    template<typename Type>
    operator Type() const {
        // get the derived TemplatedParameterEntry
        TemplatedParameterEntry<Type> * ptr = upcast<Type>();
        return ptr->value_;
    }

    //! Assigns a value of Type. Type must match the initial type otherwise runtime error is thrown.
    template<typename Type>
    ParameterEntry operator=(const Type & value){
        // get the derived TemplatedParameterEntry
        TemplatedParameterEntry<Type> * ptr = upcast<Type>();
        ptr->value_ = value;
        return *ptr;
    }

    //! Convert the stored value into a string and return it
    //! If initial type is not convertible, error is thrown
    virtual std::string repr(){
        return this->repr();
    };

    virtual ~ParameterEntry() = default;
    virtual std::string type(){
        return "Undefined";
    }
    protected:
    //! Retrieves the address of `NSL::TemplatedParameterEntry` the runtime polymorphism points to
    /*!
     * This function performs an upcast to the derived onject `template<typename Type> NSL::TemplateParameterentry`.
     * and performs checks that the conversion worked out correctly.
     * If dynamic_cast fails (returning a nullptr) a std::runtime_error is thrown
     * */
    template<typename Type>
    NSL::TemplatedParameterEntry<Type> * upcast(){
        // if conversion does not work ptr = nullptr or throw std::bad_cast
        NSL::TemplatedParameterEntry<Type> * ptr = dynamic_cast<TemplatedParameterEntry<Type> *>(this);
        
        // check for nullptr and return or throw
        if(ptr){
            return ptr;
        } else {
            std::cout << "upcast from: " << this->type() << " to: " << typeid(Type).name() << "was attempted" << std::endl;
            throw std::runtime_error("Converting ParameterEntry to TemplatedParameterEntry failed");
        }
    }
};

/*! 
 * This class implements a wrapper around a simple data type `Type` and is
 * implementing the back end for `NSL::ParameterEntry`.
 * The user most likely should not need to worry about this class as the point
 * of access is really the `NSL::ParameterEntry`
 * */
template<typename Type>
class TemplatedParameterEntry: public ParameterEntry{
    public:
    //! Construct the object given a value of type Type
    TemplatedParameterEntry(const Type & value):
        ParameterEntry(),
        value_(value)
    {}

    //! Construct the object using a default constructor
    TemplatedParameterEntry() = default;

    std::string repr() override {
        if constexpr(std::is_convertible_v<Type,std::string> || NSL::Concept::isNumber<Type>){
            return NSL::to_string(value_);
        } else {
            // e.g. NSL::SpatialLattice
            return value_.name();
        }
    }

    std::string type(){
        return typeid(value_).name();
    }
    
    //! Befriend ParameterEntry such that it can access everything in here
    friend class ParameterEntry;

    protected:
    //! The stored object
    Type value_;

};

/*!
 * A dictionary of `ParameterEntry`s references by a std::string.
 * This class is basically and `std::unordered_dict` with one overload 
 * (`operator[](const std::string&)`) and one extension `addParameter`
 * The `ParameterEntry` is a runtime polymorphism around arbitrary datatype
 * and can be added using the addParameter class.
 *
 * all std::unordered_map routines, except of `operator[]`, return a pointer to `ParameterEntry`.
 *
 * This class is strongly inspired by an old ParameterFile Management system
 * I implemented some years ago:
 * https://github.com/Marcel-Rodekamp/ParameterFileManager
 * */
class Parameter: public std::unordered_map<std::string, ParameterEntry *>{
    public:
    //! Dereference the Parameter dictionary by a std::string key.
    /*!
     * This returns a ParameterEntry object. Direct access to the stored 
     * data is not available. Use 
     * ```
     * NSL::Parameter params;
     * params.addParameter<double>("myParam")
     *
     * std::cout << params["myParam"].to<double>() << std::endl;
     * ```
     *
     * Implicit conversion however works just fine
     * ```
     * NSL::Parameter params;
     * params.addParameter<double>("myParam")
     *
     * double myParamCopy = params["myParam"];
     * ```
     * or by overwriting the value
     * ```
     * params["myParam"] = 2.;
     * ```
     * */
    ParameterEntry & operator[](const std::string & key){
        return *(std::unordered_map<std::string, ParameterEntry *>::operator[](key));
    }

    //! Add a parameter at key, default constructed value
    template<typename Type>
    void addParameter(const std::string & key){
        insert( std::make_pair<const std::string &, ParameterEntry *>(
            key,
            new TemplatedParameterEntry<Type>(Type{})
        ));
    }

    //! Add a parameter at `key` with value `value`
    template<typename Type>
    void addParameter(const std::string & key, const Type & value){
        insert( std::make_pair<const std::string &, ParameterEntry *>(
            key,
            new TemplatedParameterEntry<Type>(value)
        ));
    }
};

} //namespace NSL

#endif // NSL_PARAMETER
