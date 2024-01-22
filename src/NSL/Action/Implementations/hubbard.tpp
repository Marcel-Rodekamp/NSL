#ifndef NSL_HUBBARD_TPP
#define NSL_HUBBARD_TPP

#include "parameter.tpp"

namespace NSL::Hubbard{
enum Species{ Particle, Hole };

template<NSL::Concept::isNumber Type>
Type tilde(NSL::Parameter & params, std::string key){
    if (!params.contains("delta")) {
        NSL::size_t Nt = params["Nt"];
        Type beta = params["beta"];

        params.addParameter<Type>("delta", beta/Nt);
    }

    return Type(params["delta"])*Type(params[key]);
}
}

#endif //NSL_HUBBARD_TPP
