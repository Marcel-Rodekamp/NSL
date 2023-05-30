#ifndef NSL_COMMAND_LINE_INTERFACE
#define NSL_COMMAND_LINE_INTERFACE

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include "parameter.tpp"

namespace NSL {

class CLI{
public:
    CLI() = default;
    CLI(const CLI &) = default;
    CLI(CLI &&) = default;

    template<typename Type>
    void addArgument(const std::string & arg, bool required = false){
        NSL::Parameter<Type> p;

        options_.push_back(
            option(arg.c_str(), required_argument)
        );

        params_[arg] = &p;
    }
    
    NSL::ParameterList parse(int argc, char** argv);

protected:
    std::vector<option> options_ = {};
    NSL::ParameterList params_;
};


NSL::ParameterList NSL::CLI::parse(int argc, char** argv){
    int opt = 0;
    int idx = 0;

    while ( (opt = getopt_long(argc, argv, "", options_.data(), &idx)) != -1 ){
        switch(opt){
            case 0:
                if(optarg){
                    params_[options_[idx].name]->fromString(
                        std::string(optarg)
                    );
                }
                std::cout << "Done" << std::endl;

                break;
            case '?':
                break;
        }
    }

    return params_;
}

} // namespace NSL

#endif // NSL_COMMAND_LINE_INTERFACE

