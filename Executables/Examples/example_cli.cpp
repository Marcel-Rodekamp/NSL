#include "NSL.hpp"

/*
 * This example shows a default use case of the cli following the used library:
 * https://github.com/CLIUtils/CLI11
 *
 * Or directly look for a simple example:
 * https://github.com/CLIUtils/CLI11/blob/main/examples/simple.cpp
 * */

void printMyParams(std::string file, int runParamA, double runParamB, NSL::complex<double> runParamC){
    std::cout << "\nprinting from function" << std::endl;
    std::cout << "working with parameter file: " << file << std::endl;
    std::cout << "using run parameter A: " << runParamA << std::endl;
    std::cout << "using run parameter B: " << runParamB << std::endl;
    std::cout << "using run parameter C: " << runParamC << std::endl;
}

int main(int argc, char ** argv){
    
    // declare parameters to store the different command line obtions
    // Note, this could be simple basic types, but using a parameter class
    // might make it easier to work with throughout NSL
    NSL::Parameter params;

    // Add a parameter to store a file name 
    params.addParameter<std::string>("file");
    // Add another parameter to store an integer value
    params.addParameter<int>("runParamA");
    // Add another parameter to store a double value
    params.addParameter<double>("runParamB");
    // Add another parameter to store a complex double value
    params.addParameter<NSL::complex<double>>("runParamC");
    // Note we can extend this class in a similar way at any point

    // Declare the CLI object
    CLI::App app{"Example CLI"};
    
    // Add the different command line argument options
    // the templates specify what type to store the result in (`NSL::ParameterEntry`)
    // and what data type this corresponds to (std::string,int,double,...)
    // The first argument represents the option (see https://github.com/CLIUtils/CLI11 for more info)
    // The second argument specifies in which parameter entry the result should be stored
    // The third argument is used to print the help message accessible with -h or --help
    // Add an option to query for a file
    app.add_option<NSL::ParameterEntry,std::string>("-f, --file",  params["file"], "Provide a parameter file");
    // Add an option to query for runParamA
    app.add_option<NSL::ParameterEntry,int>("-A, --runParamA", params["runParamA"], "Specify a run parameter A");
    // Add an option to query for runParamB
    app.add_option<NSL::ParameterEntry,double>("-B, --runParamB", params["runParamB"], "Specify a run parameter B");
    // Add an option to query for runParamB
    app.add_option<NSL::ParameterEntry,NSL::complex<double>>("-C, --runParamC", params["runParamC"], "Specify a run parameter C");

    // parse the command line arguments 
    try {
        app.parse(argc,argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    // We can print the results using the NSL::ParameterEntry.to<Type>() method
    std::cout << "working with parameter file: " << params["file"].to<std::string>() << std::endl;
    std::cout << "using run parameter A: " << params["runParamA"].to<int>() << std::endl;
    std::cout << "using run parameter B: " << params["runParamB"].to<double>() << std::endl;
    std::cout << "using run parameter C: " << params["runParamC"].to<NSL::complex<double>>() << std::endl;

    // this fails at runtime as runParamB should be of type double
    //std::cout << "using run parameter B: " << params["runParamB"].to<int>() << std::endl;

    // Or implicitly convert the NSL::ParameterEntry to it's corresponding 
    // Types. Notice, also here, the type must match exactly. Currently, we
    // don't support explicit/implicit casts to other types.
    printMyParams(params["file"], params["runParamA"], params["runParamB"], params["runParamC"]);
    
    // Consequently, this would fail (runParamB is not an integer)
    //printMyParams(params["file"], params["runParamB"], params["runParamB"], params["runParamC"]);

    return EXIT_SUCCESS;
}
