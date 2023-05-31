#ifndef NSL_COMMAND_LINE_INTERFACE
#define NSL_COMMAND_LINE_INTERFACE

#include<CLI/CLI.hpp>
#include "parameter.tpp"

namespace NSL {

namespace Logger {
    //! This function adds the required arguments to the provided CLI
    /*! 
     * This functions adds the logging facility to the command line interface.
     * After parsing the command line arguments the logger can be initialized
     * using 
     * ```
     * NSL::Logger::init(
     *     log_level, // std::string
     *     log_file   // std::string 
     * );
     * ```
     * */
inline void add_logger(CLI::App & app, std::string & log_level, std::string & log_file){
    log_level = "info";
    log_file = "";

    // NSL::Logger::do_profile is a global static bool initialized in 
    // src/NSL/logger.hpp
    app.add_flag("-p, --profile", NSL::Logger::do_profile, 
        "Toggle profiling, default = false"
    );

    std::string helpMsg = "Define log level. Possible values:\n"
                          "\t- debug (prints debug information)\n"
                          "\t- info  (prints regular run information) [DEFAULT]\n"
                          "\t- warn  (prints warnings and errors only)\n"
                          "\t- error (prints errors only)";
    app.add_option("-l, --log-level",log_level, helpMsg);
    
    app.add_option("-o, --log-outfile",log_file,
        "Provide a file to put the logging to"
    );
}

} //namespace Logger

//! Initialize the logger 
/*!
 * \param argc, number of command line arguments
 * \param argv, command line argument
 * \param CLIName, name of the executable printed in the help message.
 *
 * The logger is initialized. The available command line options
 * can be found at `NSL::Logger::init`
 * */
NSL::Parameter init(int argc, char ** argv, std::string CLIName = "NSL"){
    CLI::App app{CLIName};

    // this add options to app and initializes log_level, log_file
    std::string log_level;
    std::string log_file;
    NSL::Logger::add_logger(app, log_level, log_file);

    // Define a bunch of default options 
    NSL::Parameter params;
    params.addParameter<std::string>("file");
    app.add_option<NSL::ParameterEntry, std::string>("-f, --file", params["file"], 
        "Provide a parameter file"
    );

    bool useGPU = false;
    app.add_flag("--GPU", useGPU, "Toggle GPU usage, according to this params['device'] is NSL::GPU or NSL::CPU [DEFAULT]");

    // parse the command line arguments 
    try {
        app.parse(argc,argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }

    // initialize the use of GPU/CPU. This is not restrictive but 
    // params["device"] can be used for convenience.
    if (useGPU){
        params.addParameter<NSL::Device>("device", NSL::Device("GPU"));
    } else {
        params.addParameter<NSL::Device>("device", NSL::Device("CPU"));
    }
    
    // after the logging has been parsed we can simply initialize the logger
    NSL::Logger::init(log_level, log_file);

    NSL::Logger::info(
        "Provided file: {}", std::string(params["file"])
    );

    NSL::Logger::info(
        "Using device: {}", NSL::Device(params["device"]).repr()
    );

    return params;
}

NSL::Parameter init(int argc, char ** argv, 
        std::string log_pattern,
        std::string profile_pattern,
        std::string CLIName = "NSL"
){
    CLI::App app{CLIName};

    // this add options to app and initializes log_level, log_file
    std::string log_level;
    std::string log_file;
    NSL::Logger::add_logger(app, log_level, log_file);

    // Define a bunch of default options 
    NSL::Parameter params;
    params.addParameter<std::string>("file");
    app.add_option<NSL::ParameterEntry, std::string>("-f, --file", params["file"], 
        "Provide a parameter file"
    );

    bool useGPU = false;
    app.add_flag("--GPU", useGPU, "Toggle GPU usage, according to this params['device'] is NSL::GPU or NSL::CPU [DEFAULT]");

    // parse the command line arguments 
    try {
        app.parse(argc,argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }

    // initialize the use of GPU/CPU. This is not restrictive but 
    // params["device"] can be used for convenience.
    if (useGPU){
        params.addParameter<NSL::Device>("device", NSL::Device("GPU"));
    } else {
        params.addParameter<NSL::Device>("device", NSL::Device("CPU"));
    }
    
    // after the logging has been parsed we can simply initialize the logger
    NSL::Logger::init(log_level, log_file, log_pattern, profile_pattern);

    NSL::Logger::info(
        "Provided file: {}", std::string(params["file"])
    );

    NSL::Logger::info(
        "Using device: {}", NSL::Device(params["device"]).repr()
    );

    return params;
}

} // namespace NSL

#endif // NSL_COMMAND_LINE_INTERFACE
