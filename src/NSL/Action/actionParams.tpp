#ifndef NSL_ACTION_PARAMS_TPP
#define NSL_ACTION_PARAMS_TPP

#include "concepts.hpp"
#include "typePromotion.hpp"

namespace NSL::Action {

// forward declare Base action
//template<class ParamsType, NSL::Concept::isNumber Type, NSL::Concept::isNumber ... TensorTypes> class BaseAction;
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber ... TensorTypes> class BaseAction;
    
//! Parameter class for the actions
/*
 * This class stores the possible parameters for a given action specified 
 * by `ActionType`.
 * This class is meant to be specialized for a given action found in 
 * the implementations.
 * By default it is an empty struct.
 * */
template<NSL::Concept::isTemplateDerived<BaseAction> ActionType>
struct Parameters{};


} // namespace NSL::Action 

#endif // NSL_ACTION_PARAMS_TPP
