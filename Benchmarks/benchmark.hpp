#ifndef NANOSYSTEMLIBRARY_BENCHMARK_HPP
#define NANOSYSTEMLIBRARY_BENCHMARK_HPP

#include <chrono>
#include <iostream>

template<class clock, class timeUnit>
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

        void start(){
            if(running_){
                throw std::runtime_error(std::string("StopWatch (") +  name_ + ") already running!");
            }
            running_ = true;
            start_ = clock::now();
        }

        void stop(){
            if(!running_){
//                throw std::runtime_error(std::string("StopWatch (") <<  name_ << ") not running!");
            }
            running_ = false;
            end_ = clock::now();
        }

        double get_time(double factor = 1.){
            if(running_){
                end_ = clock::now();
            }
            return std::chrono::duration_cast<timeUnit>(end_ - start_).count() * factor;
        }


};

#endif //NANOSYSTEMLIBRARY_BENCHMARK_HPP
