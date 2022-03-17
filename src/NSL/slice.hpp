#ifndef NSL_SLICE_HPP
#define NSL_SLICE_HPP

#include <cstdint>
#include <array>
#include <tuple>

#include "torch/torch.h"

namespace NSL {

struct none_t {
    none_t() = default;

    //! Static conversion to torch::indexing::None
    static const c10::nullopt_t torch() {
        return torch::indexing::None;
    }

    //! implicit conversion to torch::indexing::None
    operator c10::optional<NSL::size_t> () {
        return torch::indexing::None;
    }

    //! implicit conversion to torch::indexing::None
    operator c10::nullopt_t () {
        return torch::indexing::None;
    }
} None,End;

//! Single Index Indexer
/*!
 * Implicit conversion from a single element
 * */
class Slice {
    public:

    template<typename StartType = none_t, typename StopType = none_t, typename StepType = none_t>
    explicit Slice(StartType start, StopType stop = None, StepType step = None) {
        static_assert(std::is_same<StartType,none_t>::value || std::is_convertible_v<StartType,NSL::size_t>, "StartType must be NSL::size_t or NSL::none_t");
        static_assert(std::is_same<StopType,none_t>::value || std::is_convertible_v<StopType,NSL::size_t>, "StopType must be NSL::size_t or NSL::none_t");
        static_assert(std::is_same<StepType,none_t>::value || std::is_convertible_v<StepType,NSL::size_t>, "StepType must be NSL::size_t or NSL::none_t");

        NSL::size_t start_,stop_,step_;

        if(std::is_same<StepType,none_t>::value){step_ = 1;}
        else if constexpr(std::is_convertible_v<StepType,NSL::size_t>){
            assert(step != 0);
            step_ = step;
        }
        if(std::is_same<StartType,none_t>::value){start_ = 0;}
        else if constexpr(std::is_convertible_v<StartType,NSL::size_t>){start_ = start;}

        if(std::is_same<StopType,none_t>::value){
            stop_ = step_ < 0 ? std::numeric_limits<NSL::size_t>::min() : std::numeric_limits<NSL::size_t>::max();
        }
        else if constexpr(std::is_convertible_v<StopType,NSL::size_t>){
            stop_ = stop;
        }

        t_ = std::make_tuple(start_,stop_,step_);
    };

    Slice() : Slice(None,None,None) {}

    //explicit Slice(NSL::size_t start) : Slice(start,None,None) {}

    inline NSL::size_t start() const {return std::get<0>(t_);}
    inline NSL::size_t  stop() const {return std::get<1>(t_);}
    inline NSL::size_t  step() const {return std::get<2>(t_);}
    inline std::tuple<NSL::size_t,NSL::size_t,NSL::size_t> operator()(){
        return t_;
    };

    operator torch::indexing::Slice () const {
            return torch::indexing::Slice(std::get<0>(t_),std::get<1>(t_),std::get<2>(t_));
    }

    private:

    //! Variable to hold the start, step and stop parameters:
    /*!
     * t_[1] = start
     * t_[2] = stop
     * t_[3] = step
     */
     std::tuple<NSL::size_t,NSL::size_t,NSL::size_t> t_;

};

} // namespace NSL

#endif //NSL_SLICE_HPP
