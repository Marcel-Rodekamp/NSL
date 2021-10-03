#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

//!  \file Tensor/tensor.hpp

//! Tensor is a wrapper of torch::tensor
#include <torch/torch.h>
//! Used for handling parameter packs to reduce amount of recursive functions
#include <array>
//! If not defined NDEBUG assertion handling
#include "../assert.hpp"
//! Helper functions for complex valued Tensor
#include "../complex.hpp"
////! TimeTensor, specialized class which first dimension will (!) be parallelized across multiple nodes
//#include "time_tensor.hpp"

//! Imported Namespace: torch::indexing
using namespace torch::indexing;

namespace NSL {

//! Representation of multidimensional data.
/*!
 * Storing data of various data types is one of the key requirements of any simulation.
 * This class provides an interface to torch::Tensor ([libtorch](https://pytorch.org/cppdocs/)) in order to allow
 * access and functionality on various architectures.
 * */
template <typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
class Tensor {
    //! Alias: size_t = long int
    using size_t = long int;


    public:
        //Default constructor not required
        constexpr explicit Tensor() = delete;

        //! D-dimensional constructor.
        /*! Constructs the Tensor with D dimensions. D is determined by the number of arguments provided.\n
         * \n
         * Params:\n
         *     * `size0`: Extend of the 0th dimension\n
         *     * `sizes`: Parameter pack, extends of respective dimensions
         *
         * \n
         * Assumptions:\n
         *     * At least one argument must be passed (`size0`)
         *     * ArgsT must be of integral type (convertible to `size_t`)
         *     * Tested Types: `bool`, `float`, `double`, `NSL::complex<float>`, `NSL::complex<double>`
         *
         *
         * \n
         * Further behavior:\n
         *     * Initialization sets all values to `Type` equivalent of 0
         */
        template<typename... ArgsT>
        constexpr explicit Tensor(const size_t & size0, const ArgsT &... sizes):
            data_(torch::zeros({size0, sizes...},torch::TensorOptions().dtype<Type>()))
        {}

        //! copy constructor
        constexpr Tensor(const Tensor& other):
            data_(other.data_)
        {}

        //! move constructor
        constexpr Tensor(Tensor && other) noexcept:
            data_(std::move(other.data_))
        {}

        //! Implicit Conversion from torch::Tensor
        constexpr Tensor(torch::Tensor && other):
                data_(std::move(other))
        {}

        //! Implicit Conversion from torch::Tensor
        constexpr explicit Tensor(torch::Tensor & other):
                data_(other)
        {}

        // =====================================================================
        // Tensor Creation Helpers
        // =====================================================================

        //! Fill the tensor with pseudo-random numbers
        /*!
         * Fills the Tensor with pseudo-random numbers from the uniform distribution
         * \todo Generalize for different distributions
         */
        Tensor<Type,RealType> & rand(){
            this->data_.uniform_();
            return *this;
        }

        // =====================================================================
        // Accessors
        // =====================================================================

        //! Random Access operator
        /*!
         *  * `const Args &... indices`: parameter pack, indices of the tensor
         *      * Args must be integer type e.g.: `int`, `size_t`,...
         *
         *  \n
         *  Behavior:\n
         *  each index in parameter pack `indices` corresponds to the index
         *
         * */
        template<typename ...Args>
        constexpr Type & operator()(const Args &... indices) {
            // check that all arguments of the parameter pack are convertible to
            // the defined size type (i.e. integer valued)
            static_assert(NSL::all_convertible<size_t, Args...>::value);

            return data_.data_ptr<Type>()[linearIndex_(indices...)];
        }

        template<typename ...Args>
        constexpr Type & operator()(const Args &... indices) const {
            static_assert(NSL::all_convertible<size_t, Args...>::value,
                          "NSL::Tensor::operator()(const Args &... indices) can only be called with arguments of integer type"
            ); // static_assert
            // need data_.dim() arguments!
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const Args &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const Args &... indices) called with to many indices");

            // ToDo: data_.dim() == 1 is a problem as the slice case would always be called! Unfortunately, data_.dim() is only known at runtime
            return data_.data_ptr<Type>()[linearIndex_(indices...)];

        }

        friend torch::Tensor to_torch(const Tensor<Type> & other) {
            return other.data_;
        }

        constexpr Type * data(){
            return data_.data_ptr<Type>();
        }

        constexpr Type * data() const {
            return data_.data_ptr<Type>();
        }


        // =====================================================================
        // Slice Operation
        // =====================================================================

        constexpr Tensor<Type,RealType> & slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step = 1){
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return std::move(Tensor<Type>(slice));
        }

        constexpr Tensor<Type,RealType> & slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step = 1) const {
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return Tensor<Type>(slice);
        }


        // =====================================================================
        // Assignment operators
        // =====================================================================

        //! Assignment operator: Tensor to Tensor
        Tensor<Type,RealType> & operator=(const Tensor<Type> & other){
            // deep copy of other into this
            this->data_ = other.data_.clone();
            return *this;
        }

        //! Assignment operator: Scalar to Tensor
        Tensor<Type,RealType> & operator=(const Type & value){
            // overwrite data with value
            this->data_ = torch::full_like(this->data_, value);
            return *this;
        }


        // =====================================================================
        // Boolean operators
        // =====================================================================

        //! Elementwise equal: Tensor to Tensor
        /*
         * Checks if each element of this equals other.
         * */
        Tensor<bool> operator== (NSL::Tensor<Type> & other) {
            return this->data_ == other.data_;
        }

        //! Elementwise equal: Tensor to number
        Tensor<bool> operator== (const Type & value) {
            return this->data_ == value;
        }

        //! Elementwise not equal: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator!= (const NSL::Tensor<Type,RealType> & other) {
            return this->data_ != other.data_;
        }

        //! Elementwise not equal: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator!= (const Type & value) {
            return this->data_ != value;
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator<= (const NSL::Tensor<Type,RealType> & other) {
            return this->data_ <= other.data_;
        }

        //! Elementwise smaller or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator<= (const Type & value) {
            return this->data_ <= value;
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator>= (const NSL::Tensor<Type,RealType> & other) {
            return this->data_ >= other.data_;
        }

        //! Elementwise greater or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator>= (const Type & value) {
            return this->data_ >= value;
        }

        //! Elementwise smaller : Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator< (const NSL::Tensor<Type,RealType> & other) {
            return this->data_ < other.data_;
        }

        //! Elementwise smaller: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator< (const Type & value) {
            return this->data_ < value;
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator> (const NSL::Tensor<Type,RealType> & other) {
            return this->data_ > other.data_;
        }

        //! Elementwise greater or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator> (const Type & value) {
            return this->data_ > value;
        }

        // =====================================================================
        // Print and stream
        // =====================================================================

        //! Streaming operator
        //! \todo: We should code up our own.
        friend std::ostream & operator << (std::ostream & os, const Tensor<Type> & T){
            return (os << T.data_);
        }

        //Print tensor.
        void print () const{
            std::cout<<data_ <<std::endl;
        }

        //Print tensor complex.
        void print_complex () const{
            std::cout<< torch::view_as_real(data_)<<std::endl;
        }

        // =====================================================================
        // Shape and dimension
        // =====================================================================

        //! Get the extent of a certain dimension.
        /*!
         * Parameters:\n
         * * `const size_t & dim`: Dimension of which the extent should be queried.
         *
         * Behavior:\n
         * Returns the extent of the dimension specified by `dim`.
         * If no reallocation is performed the value will match the given parameter
         * to the constructor `NSL::Tensor<Type,RealType>::Tensor(Arg size0, Args... sizes)`.
         * */
        [[nodiscard]] std::size_t shape(const size_t & dim) const {
            return data_.size(dim);
        }

        //! Get the extents of the tensor
        [[nodiscard]] std::vector<size_t> shape() const {
        std::vector<long int> out(data_.dim());
        for (long int i=0; i< data_.dim(); ++i){
            out[i] = data_.size(i);
        }
        return out;
    }

        //! Get the dimension of the Tensor.
        /*!
         *  The dimension of the tensor is specified at construction by the number
         *  of integer arguments provided to the constructor `NSL::Tensor(Arg size0, Args... sizes)`
         * */
        [[nodiscard]] std::size_t dim() const {
            return this->data_.dim();
        }


        // =====================================================================
        // Algebra Operators
        // =====================================================================
        // Here we create all the inplace linear algebra operation we need

        // =====================================================================
        // operator+;

        //! Elementwise addition: Tensor + Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator+(const Tensor<Type,RealType> & other){
            Tensor<Type,RealType> tmp(this->data_ + other.data_);
            return tmp;
        }

        //! Elementwise addition: Tensor + number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator+(const Type & value){
            Tensor<Type,RealType> tmp(this->data_ + value);
            return tmp;
        }

        // =====================================================================
        // operator+=;

        //! Elementwise addition: Tensor + Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> & operator+=(const Tensor<Type,RealType> & other){
            this->data_ += other.data_;
            return *this;
        }

        //! Elementwise addition: Tensor + number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator+=(const Type & value){
            this->data_ += value;
            return *this;
        }


        // =====================================================================
        // operator-;

        //! Elementwise subtraction: Tensor - Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator-(const Tensor<Type,RealType> & other){
            Tensor<Type,RealType> tmp(this->data_ - other.data_);
            return tmp;
        }

        //! Elementwise subtraction: Tensor - number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator-(const Type & value){
            Tensor<Type,RealType> tmp(this->data_ - value);
            return tmp;
        }

        // =====================================================================
        // operator-=;

        //! Elementwise subtraction: Tensor - Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> & operator-=(const Tensor<Type,RealType> & other){
            this->data_ -= other.data_;
            return *this;
        }

        //! Elementwise subtraction: Tensor - number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator-=(const Type & value){
            this->data_ -= value;
            return *this;
        }


        // =====================================================================
        // operator*;

        //! Elementwise multiplication (Schur,Hadamard product): Tensor * Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator*(const Tensor<Type,RealType> & other){
            Tensor<Type,RealType> tmp(this->data_ * other.data_);
            return tmp;
        }

        //! Elementwise multiplication: Tensor * number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator*(const Type & value){
            Tensor<Type,RealType> tmp(this->data_ * value);
            return tmp;
        }

        // =====================================================================
        // operator*=;

        //! Elementwise multiplication (Schur,Hadamard product): Tensor * Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> & operator*=(const Tensor<Type,RealType> & other){
            this->data_ *= other.data_;
            return *this;
        }

        //! Elementwise multiplication: Tensor * number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator*=(const Type & value){
            this->data_ *= value;
            return *this;
        }


        // =====================================================================
        // operator/;

        //! Elementwise division (Schur, Hadamard division): Tensor / Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator/(const Tensor<Type,RealType> & other){
            Tensor<Type,RealType> tmp(this->data_ / other.data_);
            return tmp;
        }

        //! Elementwise division: Tensor / number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator/(const Type & value){
            Tensor<Type,RealType> tmp(this->data_ / value);
            return tmp;
        }

        // =====================================================================
        // operator/=;

        //! Elementwise division (Schur, Hadamard division): Tensor / Tensor
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> & operator/=(const Tensor<Type,RealType> & other){
            this->data_ /= other.data_;
            return *this;
        }

        //! Elementwise division: Tensor / number
        /*!
         * \todo Add documentation.
         */
        Tensor<Type,RealType> operator/=(const Type & value){
            this->data_ /= value;
            return *this;
        }

        //! Elementwise real part
        /*!
         * If `Type` refers to `NSL::complex<RealType>` return the real part of
         * each element of the tensor as `NSL::Tensor<RealType>`
         * Else `Type` refers to a RealType expression (e.g. `float`,`double`, ...)
         * and is simply returned.
         * */
        Tensor<RealType,RealType> real(){
            // compile time evaluation: if Type equals RealType
            if constexpr(std::is_same<Type,RealType>()){
                // return the real tensor
                return *this;
                // compile time evaluation: if Type equals NSL::complex<RealType>
            } else {
                // return real part
                return torch::real(this->data_);
            }
        }

        //! Elementwise imaginary part
        /*!
         * If `Type` refers to `NSL::complex<RealType>` return the imaginary part of
         * each element of the tensor as `NSL::Tensor<RealType>`
         * Else `Type` refers to a RealType expression and does not have an imaginary
         * part, a Tensor with zeros is returned.
         * */
        Tensor<RealType,RealType> imag(){
            // compile time evaluation: if type T is real type RT
            if constexpr(std::is_same<Type,RealType>()){
                // return zeros
                return torch::zeros(this->data_.sizes(),
                                    torch::TensorOptions().dtype<RealType>()
                                                          .layout(this->data_.layout())
                                                          .device(this->data_.device())
                );
                // compile time evaluation: if type T is NSL::complex<RT>
            } else {
                // return imaginary part
                return torch::imag(this->data_);
            }
        }

        // =====================================================================
        // Trigonometric functions
        // =====================================================================

        //! Elementwise exponential
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & exp() {
            this->data_.exp_();
            return *this;
        }

        //! Elementwise sine
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & sin() {
            this->data_.sin_();
            return *this;
        }

        //! Elementwise cosine
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & cos() {
            this->data_.cos_();
            return *this;
        }

        //! Elementwise tangent
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & tan() {
            this->data_.tan_();
            return *this;
        }

        //! Elementwise hyperbolic sine
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & sinh() {
            this->data_.sinh_();
            return *this;
        }

        //! Elementwise hyperbolic cosine
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & cosh() {
            this->data_.cosh_();
            return *this;
        }

        //! Elementwise hyperbolic tangent
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> & tanh() {
            this->data_.tanh_();
            return *this;
        }


        // =====================================================================
        // Reductions
        // =====================================================================

        //! Reduction: +
        /*! \todo: Add Documentation*/
        Type sum(const size_t dim){
            return this->data_.sum(dim).template item<Type>();
        }

        //! Reduction: +
        /*! \todo: Add Documentation*/
        Type sum(){
            return this->data_.sum().template item<Type>();
        }

        //! Reduction: *
        /*! \todo: Add Documentation*/
        Type prod(const size_t dim){
            return this->data_.prod(dim).template item<Type>();
        }

        //! Reduction: *
        /*! \todo: Add Documentation*/
        Type prod(){
            return this->data_.prod().template item<Type>();
        }

        //! Reduction: && (logical and)
        /*! \todo: Add Documentation*/
        Type all(){
            static_assert(std::is_same<Type,bool>());
            return this->data_.all().template item<Type>();
        }

        //! Reduction: || (logical or)
        /*! \todo: Add Documentation*/
        Type any(){
            static_assert(std::is_same<Type,bool>());
            return this->data_.any().template item<Type>();
        }


        // =====================================================================
        // Resize operations
        // =====================================================================

        //! Expanding the Tensor by one dimension with size `newSize`
        /*! \todo: Add Documentation
         * */
        NSL::Tensor<Type> & expand(const size_t & newSize) {
            this->data_ = data_.unsqueeze(-1).expand(c10::IntArrayRef({this->data_.sizes(), newSize}));
            return *this;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const int shift){
            this->data_ = this->data_.roll(shift,0);
            return *this;
        }

        //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const int shift, const size_t dim){
            this->data_ = this->data_.roll(shift,dim);
            return *this;
        }

        //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const int shift, const Type boundary){
            this->data_ = this->data_.roll(shift,0);

            if(shift>0){
                this->data_.slice(/*dim=*/0,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
            } else {
                this->data_.slice(/*dim=*/0,/*start=*/this->shape(0)-shift,/*end=*/this->shape(0),/*step=*/1)*=boundary;
            }

            return *this;
        }

        //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const int shift, const size_t dim, const Type boundary){
            this->data_ = this->data_.roll(shift,dim);

            if(shift>0){
                this->data_.slice(/*dim=*/dim,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
            } else {
                this->data_.slice(/*dim=*/dim,/*start=*/this->shape(dim)-shift,/*end=*/this->shape(dim),/*step=*/1)*=boundary;
            }

            return *this;
        }

    protected:
        //! Underlying torch::Tensor
        torch::Tensor data_;

        //! Transform a given set of D indices to the linear index used to reference
        //! the 1 dimensional memory layout
        template<typename... Args>
        inline std::size_t linearIndex_(const Args &... indices){
            // check that the number of arguments in indices matches the dimension of the tensor
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const Args &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const Args &... indices) called with to many indices");

            // unpack the parameter pack
            std::array<long int, sizeof...(indices)> a_indices = {indices...};

            size_t offset = 0;
            for(size_t d = 0 ; d < sizeof...(indices); ++d){
                offset += a_indices[d] * data_.stride(d);
            }

            return offset;
        }

}; //Tensor class


template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
using TimeTensor = NSL::Tensor<Type>;


} // namespace NSL

#endif //NSL_TENSOR_HPP
