#include "NSL.hpp"

/*
 * This example shows a default use case of the cli following the used library:
 * https://github.com/CLIUtils/CLI11
 *
 * Or directly look for a simple example:
 * https://github.com/CLIUtils/CLI11/blob/main/examples/simple.cpp
 * */

void printMyParams(std::string file, int runParamA, double runParamB, NSL::complex<double> runParamC){
    NSL::Logger::info("printMyParams: working with parameter file: {}", file);
    NSL::Logger::info("printMyParams: using run parameter A: {}", runParamA);
    NSL::Logger::info("printMyParams: using run parameter B: {}", runParamB);
    NSL::Logger::info("printMyParams: using run parameter C: {}", NSL::to_string(runParamC));
}

int main(int argc, char ** argv){
    // This is a very compact form to initialize the NSL::Logger as well as
    // a parameter class from the command line.
    // Basically, the command line is parsed for the parameters in the first
    // initializer list, i.e. it is looking for 
    // --file STR
    // --runParamA INT
    // --runParamB DOUBLE
    // --runParamC COMPLEX
    // the data types are provided with the template parameters matching 
    // the order of the parameter strings. 
    // The second initializer list provides a set of help messages that 
    // are displayed using --help or -h. If omitted messages will be empty.
    // The last argument (optional) provides a name to the executable which
    // is printed for the help message
    NSL::Parameter params = NSL::init<std::string, int, double, NSL::complex<double>>(
        argc,argv,
        {"file","runParamA","runParamB","runParamC"},
        {"Provide a parameter file", "Specify a run parameter A",
         "Specify a run parameter B", "Specify a run parameter C, format: '(real,imag)'"},
        "Example CLI"
    );
    // if no parameters have to be passed the minimal initialization should be
    // NSL::init(argc,argv); // returns void
    // This initializes the logger only
    // for a more sophisticated ansatz with more freedom, please refer to 
    // the command at the end of this file showing more detail.

    // We can print the results using the NSL::ParameterEntry.to<Type>() method
    NSL::Logger::info("working with parameter file: {}", params["file"].to<std::string>());
    NSL::Logger::info("using run parameter A: {}", params["runParamA"].to<int>());
    NSL::Logger::info("using run parameter B: {}", params["runParamB"].to<double>());
    NSL::Logger::info("using run parameter C: {}", NSL::to_string(params["runParamC"].to<NSL::complex<double>>()));

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


/* This example uses a more flexible directly from the implementation of 
 * the CLI: https://github.com/CLIUtils/CLI11
 *
 * // Generate the parameter class
 * NSL::Parameter params;
 *
 * // Add a parameter to store a file name 
 * params.addParameter<std::string>("file");
 * // Add another parameter to store an integer value
 * params.addParameter<int>("runParamA");
 * // Add another parameter to store a double value
 * params.addParameter<double>("runParamB");
 * // Add another parameter to store a complex double value
 * params.addParameter<NSL::complex<double>>("runParamC");
 * // Note we can extend this class in a similar way at any point
 * // Or even change the parameters using e.g.
 * // params["runParamA"] = 3;
 * 
 * // Declare the CLI object
 * CLI::App app{"Example CLI"};
 *
 * // Add the logger arguments to the CLI
 * // this add options to app and initializes log_level, log_file
 * std::string log_level;
 * std::string log_file;
 * NSL::Logger::add_logger(app, log_level, log_file);
 * // alternatively use:
 * // params.addParameter<std::string>("log level");
 * // params.addParameter<std::string>("log file");
 * // NSL::Logger::add_logger(app, params["log level"], params["log file"]);
 *  
 * // Add the different command line argument options
 * // the templates specify what type to store the result in (`NSL::ParameterEntry`)
 * // and what data type this corresponds to (std::string,int,double,...)
 * // The first argument represents the option (see  for more info)
 * // The second argument specifies in which parameter entry the result should be stored
 * // The third argument is used to print the help message accessible with -h or --help
 * // Add an option to query for a file
 * app.add_option<NSL::ParameterEntry,std::string>("-f, --file",  params["file"], "Provide a parameter file");
 * // Add an option to query for runParamA
 * app.add_option<NSL::ParameterEntry,int>("-A, --runParamA", params["runParamA"], "Specify a run parameter A");
 * // Add an option to query for runParamB
 * app.add_option<NSL::ParameterEntry,double>("-B, --runParamB", params["runParamB"], "Specify a run parameter B");
 * // Add an option to query for runParamB
 * app.add_option<NSL::ParameterEntry,NSL::complex<double>>("-C, --runParamC", params["runParamC"], "Specify a run parameter C");
 * 
 * // parse the command line arguments 
 * try {
 *     app.parse(argc,argv);
 * } catch (const CLI::ParseError &e) {
 *     exit(app.exit(e));
 * }
 *
 * // After parsing the logger needs to be initialized with the found parameters:
 * NSL::Logger::init(log_level, log_file);
 * // again for the alternative with params
 * // NSL::Logger::init(params["log level"], params["log file"]);
 *
 * // Now everything is set up and we can proceed with the program
*/
