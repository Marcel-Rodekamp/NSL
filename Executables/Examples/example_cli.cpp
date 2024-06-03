#include "NSL.hpp"

int main(int argc, char ** argv){
    // define the general type for convinence
    typedef NSL::complex<double> Type;

    // This line parses the command line options 
    // -f, --file A parameter file
    // --GPU      A flag to to set the device
    // The results are stored in the NSL::Parameter object and can be accessed
    // with the keys "file" and "device" respectively.
    // Furthermore, the logger is initialized and the command line is
    // parsed for the logging options. For more information refer to
    // NSL::Logger::init and NSL::Logger::add_logger
    // For a more sophisticated CLI refer to the documentation at the end
    // of this file
    NSL::Parameter params = NSL::init(argc,argv,"Example CLI");
    // We can read in the parameter file and put the read data into the 
    // params object, notice this uses the example_param.yml file
    // For personal files, this code needs to be adjusted accordingly
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // convert the data from example_param.yml and put it into the params
    // The name of the physical system
    params["name"] = yml["system"]["name"].as<std::string>();
    params["beta"] = yml["system"]["beta"].as<double>();
    params["Nt"] = yml["system"]["Nt"].as<NSL::size_t>();
    params["U"] = yml["system"]["U"].as<double>();
    params["save frequency"] = yml["HMC"]["save frequency"].as<NSL::size_t>();
    params["Ntherm"] = yml["HMC"]["Ntherm"].as<NSL::size_t>();
    params["Nconf"] = yml["HMC"]["Nconf"].as<NSL::size_t>();
    params["trajectory length"] = yml["Leapfrog"]["trajectory length"].as<double>();
    params["Nmd"] = yml["Leapfrog"]["Nmd"].as<NSL::size_t>();
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();

    // Now we want to log the found parameters
    // - key is a std::string name,beta,...
    // - value is a ParameterEntry * which is a wrapper around the actual 
    //   value of interest, we can use ParameterEntry::repr() to get a string
    //   representation of the stored value
    for(auto [key, value]: params){
        NSL::Logger::info( "{}: {}", key, value);
    }

}

/* This example uses a more flexible directly from the implementation of 
 * the CLI: https://github.com/CLIUtils/CLI11 
 * and augmented to work with NSL::Parameter objects
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
