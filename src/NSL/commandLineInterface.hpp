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

//! Initialize the logger and a `NSL::Parameter`
/*!
 * \param argc, number of command line arguments
 * \param argv, command line argument
 * \param options, array of strings used as arguments e.g. file -> command line argument --file.
 *        Must match the order of template arguments 
 * \param helpMessages, array of strings used to display messages if executable is invoked with --help or -h 
 * \param CLIName, name of the executable printed in the help message.
 *
 * This initialization takes the options and puts it into a CLI. Then the 
 * command line is parsed for the provided options and results are put into 
 * an `NSL::Parameter` class. 
 * Further, the logger is initialized. The available command line options
 * can be found at `NSL::Logger::init`
 * */
template<typename ... Types>
NSL::Parameter init(
    int argc, char ** argv, 
    std::array<std::string, sizeof...(Types)> options, 
    std::array<std::string, sizeof...(Types)> helpMessages,
    std::string CLIName = "NSL"
){

    NSL::Parameter params;
    CLI::App app{CLIName};

    // this add options to app and initializes log_level, log_file
    std::string log_level;
    std::string log_file;
    NSL::Logger::add_logger(app, log_level, log_file);

    // Add the different parameters to the parameter class
    std::apply(
        [&params,&app](auto ... args) {
            ((params.addParameter<Types>(args)), ...);
        }, 
        options
    );
    
    // Add the different command line argument options
    // the templates specify what type to store the result in (`NSL::ParameterEntry`)
    // and what data type this corresponds to (Types,...)
    // The first argument represents the option (see https://github.com/CLIUtils/CLI11 for more info)
    // The second argument specifies in which parameter entry the result should be stored
    // The third argument is used to print the help message accessible with -h or --help
    // Add an option to query for a file
    int i = 0;
    std::apply(
        [&params,&app,&helpMessages,&i](auto ... args) {
            ((app.add_option<NSL::ParameterEntry, Types>(
                fmt::format("--{}",args), params[args], helpMessages[i]
            ), ++i), ...);
        },
        options
    );

    // parse the command line arguments 
    try {
        app.parse(argc,argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }

    // after the logging has been parsed we can simply initialize the logger
    NSL::Logger::init(log_level, log_file);

    // return the results
    return params;
}

//! Initialize the logger and a `NSL::Parameter`
/*!
 * \param argc, number of command line arguments
 * \param argv, command line argument
 * \param options, array of strings used as arguments e.g. file -> command line argument --file.
 *        Must match the order of template arguments 
 * \param CLIName, name of the executable printed in the help message.
 *
 * This initialization takes the options and puts it into a CLI. Then the 
 * command line is parsed for the provided options and results are put into 
 * an `NSL::Parameter` class. This overload ignores the help messages and 
 * puts empty strings instead.
 * Further, the logger is initialized. The available command line options
 * can be found at `NSL::Logger::init`
 * */
template<typename ... Types>
NSL::Parameter init(
    int argc, char ** argv, 
    std::array<std::string, sizeof...(Types)> options, 
    std::string CLIName = "NSL"
){
    return NSL::init<Types...>( argc,argv,options,std::array<std::string,sizeof...(Types)>(),CLIName );
}

//! Initialize the logger 
/*!
 * \param argc, number of command line arguments
 * \param argv, command line argument
 * \param CLIName, name of the executable printed in the help message.
 *
 * The logger is initialized. The available command line options
 * can be found at `NSL::Logger::init`
 * */
void init(int argc, char ** argv, std::string CLIName = "NSL"){
    CLI::App app{CLIName};

    // this add options to app and initializes log_level, log_file
    std::string log_level;
    std::string log_file;
    NSL::Logger::add_logger(app, log_level, log_file);

    // parse the command line arguments 
    try {
        app.parse(argc,argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }
    
    // after the logging has been parsed we can simply initialize the logger
    NSL::Logger::init(log_level, log_file);
}

} // namespace NSL

#endif // NSL_COMMAND_LINE_INTERFACE
