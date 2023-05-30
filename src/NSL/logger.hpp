#ifndef NSL_LOGGER_INCLUDE_HPP
#define NSL_LOGGER_INCLUDE_HPP

#include <iostream>
#include <unistd.h>
#include <map>
#include <utility>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/stopwatch.h>

#include<CLI/CLI.hpp>

/*! \file logger.hpp
 *  Utilities for logging to console and files
 *
 *  The main strategy is to have a singleton that handles parameters form 
 *  command line and sets up the appropriate logger.
 *    
 **/

namespace NSL::Logger {
    
    static bool do_profile = false;

    // this is a legacy functions and should be deleted better use 
    // NSL::Logger::init(std::string log_level, std::string log_file)
    // below. This can be combined with 
    // NSL::Logger::add_logger(CLI::App, std::string & log_level, std::string & log_file)
    // defined in src/NSL/commandLineInterface.hpp  to handle CLI.
    inline void init_logger(int argc, char* argv[]){
        int opt;
        std::string log_level = "info";
        std::string log_file = "";

        std::map<std::string, spdlog::level::level_enum> levels = {
            {"debug", spdlog::level::debug},
            {"info", spdlog::level::info},
            {"warn", spdlog::level::warn},
            {"error", spdlog::level::err},
        };

        while ((opt = getopt (argc, argv, "l:o:p")) != -1){
            switch (opt){
                case 'p':
                    do_profile = true;
                    break;
                case 'l':
                    log_level = std::string(optarg);
                    if(!levels.contains(log_level)) {   
                        std::cerr << "Improper usage of -l flag. Expected one of:\n";
                        std::cerr << "- debug (prints debug information)\n";
                        std::cerr << "- info  (prints regular run information) [DEFAULT]\n";
                        std::cerr << "- warn  (prints warnings and errors only)\n";
                        std::cerr << "- error (prints errors only)\n";
                    }
                    break;
                case 'o':
                    log_file = std::string(optarg);
                    break;
                case '?':
                    if (optopt == 'l'){   
                        std::cerr << "Improper usage of -l flag. Expected one of:\n";
                        std::cerr << "- debug (prints debug information)\n";
                        std::cerr << "- info  (prints regular run information) [DEFAULT]\n";
                        std::cerr << "- warn  (prints warnings and errors only)\n";
                        std::cerr << "- error (prints errors only)\n";
                    }
                    else if (optopt == 'o'){   
                        std::cerr << "Improper usage of -o flag. Expected output log filename:\n";
                    }
                    else isprint (optopt);
                    exit(1);
                default:
                    abort ();
            }
        }

        std::vector<spdlog::sink_ptr> sinks;
        if(log_file != ""){
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
            file_sink->set_level(levels["debug"]);
            sinks.push_back(file_sink);
        }

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(levels[log_level]);
        sinks.push_back(console_sink);

        auto logger = std::make_shared<spdlog::logger>("NSL_logger", begin(sinks), end(sinks));
        logger->set_pattern("[%D %T] [%l] %v");
        logger->set_level(spdlog::level::debug);
        logger->flush_on(spdlog::level::debug);

        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        if(do_profile){
            auto profile_logger = spdlog::basic_logger_st("NSL_profiler", "NSL_profile.log", true);
            profile_logger->set_pattern("[%D %T] %v");
            profile_logger->flush_on(spdlog::level::debug);
        }
    }

    //! Initialize the logger
    /*! 
     * This function initializes the logger. 
     * The parameter `log_level` must be either 
     *  - "debug" (prints debug information)
     *  - "info"  (prints regular run information) [DEFAULT]
     *  - "warn"  (prints warnings and errors only)
     *  - "error" (prints errors only)
     *  The parameter `log_file` can be used to put the log output into
     *  the specified file. If `log_file=""` output is logged to the console.
     * */
    inline void init(std::string log_level, std::string log_file){
        std::map<std::string, spdlog::level::level_enum> levels = {
            {"debug", spdlog::level::debug},
            {"info", spdlog::level::info},
            {"warn", spdlog::level::warn},
            {"error", spdlog::level::err},
        };

        std::vector<spdlog::sink_ptr> sinks;
        if(log_file != ""){
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
            file_sink->set_level(levels["debug"]);
            sinks.push_back(file_sink);
        }

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(levels[log_level]);
        sinks.push_back(console_sink);

        auto logger = std::make_shared<spdlog::logger>("NSL_logger", begin(sinks), end(sinks));
        logger->set_pattern("[%D %T] [%l] %v");
        logger->set_level(spdlog::level::debug);
        logger->flush_on(spdlog::level::debug);

        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        if(do_profile){
            auto profile_logger = spdlog::basic_logger_st("NSL_profiler", "NSL_profile.log", true);
            profile_logger->set_pattern("[%D %T] %v");
            profile_logger->flush_on(spdlog::level::debug);
        }
    }

    template <typename... Args>
    inline void debug(fmt::format_string<Args...> fmt, Args&&... args){
        spdlog::debug(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void info(fmt::format_string<Args...> fmt, Args&&... args){
        spdlog::info(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void warn(fmt::format_string<Args...> fmt, Args&&... args){
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void error(fmt::format_string<Args...> fmt, Args&&... args){
        spdlog::error(fmt, std::forward<Args>(args)...);
    }

    inline std::pair<spdlog::stopwatch, std::string> start_profile(const std::string& tag){
        if(do_profile){
            spdlog::get("NSL_profiler")->warn("Entering {}", tag);
        }
        return std::make_pair(spdlog::stopwatch(), tag);
    }

    inline void elapsed_profile(const std::pair<spdlog::stopwatch, std::string>& sw){
        if(do_profile){
            spdlog::get("NSL_profiler")->warn("Time since start of {}: {:.3} s", sw.second, sw.first);
        }
    }

    inline void stop_profile(const std::pair<spdlog::stopwatch, std::string>& sw){
        if(do_profile){
            spdlog::get("NSL_profiler")->warn("Total time spent in {}: {:.3} s", sw.second, sw.first);
        }
    }


} // namespace NSL::Logger


#endif
