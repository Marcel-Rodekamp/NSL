#ifndef NSL_BENCHMARK_HPP
#define NSL_BENCHMARK_HPP 

#include <chrono>
#include <iostream>

//! \file Benchmarks/benchmark.hpp


//! Timer class
/*!
 * `clock`: Any chrono clock can be used, by default a monotonic clock is used.
 * `timeUnit`: Specify the unit in which the time is returned. Possible choices:
 *      - https://en.cppreference.com/w/cpp/chrono/duration
 *      - std::chrono::nanoseconds
 *      - std::chrono::microseconds 
 *      - std::chrono::milliseconds
 * */
template<class clock = std::chrono::steady_clock, class timeUnit = std::chrono::seconds>
class Timer{
    private:
    std::chrono::time_point<clock> start_;
    std::chrono::time_point<clock> end_;
    bool running_;
    std::string name_;

    public:
        Timer(const std::string & name = "timer"):
            start_(clock::now()),
            end_(clock::now()),
            running_(false),
            name_(name)
        {}
        
        //! Start the timer
        void start(){
            if(running_){
                throw std::runtime_error(std::string("StopWatch (") +  name_ + ") already running!");
            }
            running_ = true;
            start_ = clock::now();
        }

        //! Stop the timer
        void stop(){
            if(!running_){
                throw std::runtime_error(std::string("StopWatch (") +  name_ + ") not running!");
            }
            running_ = false;
            end_ = clock::now();
        }

        //! Get the time in `[timeUnit]`
        double get_time(double factor = 1.){
            if(running_){
                end_ = clock::now();
            }
            return std::chrono::duration_cast<timeUnit>(end_ - start_).count() * factor;
        }
};

#endif // NSL_BENCHMARK_HPP
