#ifndef NSL_MEASUREMENTS_HPP
#define NSL_MEASUREMENTS_HPP

namespace NSL{

class Measurement{
    public:
        Measurement(NSL::Parameter params, NSL::H5IO & h5):
            params_(params),
            h5_(h5)
        {}

        //! The basic interface to call a measure routine. 
        /*!
         * All measure routines should read data from the h5_ file 
         * thus this function is independent of any arguments.
         * */
        virtual void measure() = 0;

    protected:
        NSL::Parameter params_;
        NSL::H5IO & h5_;
};
}

#endif // NSL_MEASUREMENTS_HPP

