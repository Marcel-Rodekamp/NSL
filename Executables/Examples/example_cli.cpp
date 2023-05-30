#include "NSL.hpp"

int main(int argc, char ** argv){

    NSL::CLI cli;
    cli.addArgument<int>("param1");
    cli.addArgument<float>("param2");

    auto params = cli.parse(argc,argv);

    std::cout << params["param1"]->to<int>()
              << std::endl;
    std::cout << params["param2"]->to<float>()
              << std::endl;


    return EXIT_SUCCESS;
}
