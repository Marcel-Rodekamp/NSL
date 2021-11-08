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
    using size_t = int64_t;


    public:
        //Default constructor not required
        constexpr explicit Tensor() :
            data_(torch::zeros({0},torch::TensorOptions().dtype<Type>()))
        {}

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
         *     * SizeType must be of integral type (convertible to `size_t`)
         *     * Tested Types: `bool`, `float`, `double`, `NSL::complex<float>`, `NSL::complex<double>`
         *
         *
         * \n
         * Further behavior:\n
         *     * Initialization sets all values to `Type` equivalent of 0
         */
        template<typename... SizeType>
        constexpr explicit Tensor(const size_t & size0, const SizeType &... sizes):
            data_(torch::zeros({size0, sizes...},torch::TensorOptions().dtype<Type>()))
        {}

        /*!
         * param shape a std::vector giving the shape of the new Tensor.
         **/
        explicit Tensor(const std::vector<size_t> &shape):
            data_(torch::zeros(torch::IntArrayRef(shape.data(), shape.size()),torch::TensorOptions().dtype<Type>()))
        {}

        //! copy constructor
        constexpr Tensor(const Tensor<Type,RealType>& other):
            data_(other.data_)
        {}


        //! copy constructor
        template<typename OtherRealType>
        constexpr Tensor(const Tensor<OtherRealType,OtherRealType>& other):
                data_(to_torch(other).to(torch::dtype<NSL::complex<OtherRealType>>()))
        {
            static_assert(std::is_same<OtherRealType,RealType>());
            static_assert(NSL::is_complex<Type>());
        }

        //! move constructor
        constexpr Tensor(Tensor && other) noexcept:
            data_(std::move(other.data_))
        {}

        explicit Tensor(torch::Tensor other):
            data_(std::move(other))
        {}

        explicit constexpr Tensor(torch::Tensor & other):
            data_(other)
        {}

        operator torch::Tensor(){
            return data_;
        }


        // =====================================================================
        // Tensor Creation Helpers
        // =====================================================================

        //! Fill the tensor with pseudo-random numbers
        /*!
         * Fills the Tensor with pseudo-random numbers from the uniform distribution
         * \todo Generalize for different distributions
         */
        Tensor<Type,RealType> & rand(){
            // Note: This should be a uniform U(0,1) distribution
            this->data_.uniform_();
            return *this;
        }


        //! Fill the tensor with pseudo-random numbers
        /*!
         * Fills the Tensor with pseudo-random numbers from the standard normal distribution
         * \todo Generalize for different distributions
         */
        Tensor<Type,RealType> & randn(){
            // Note: This should be a standard normal N(mean=0,var=1) distribution
            this->data_.normal_();
            return *this;
        }


        // =====================================================================
        // Accessors
        // =====================================================================

        //! Random Access operator
        /*!
         *  * `const SizeType &... indices`: parameter pack, indices of the tensor
         *      * SizeType must be integer type e.g.: `int`, `size_t`,...
         *
         *  \n
         *  Behavior:\n
         *  each index in parameter pack `indices` corresponds to the index
         *
         * */
        template<typename ...SizeType>
        constexpr Type & operator()(const SizeType &... indices) {
            return this->data_.template data_ptr<Type>()[this->linearIndex_(indices...)];
        }

        //! Random Access operator
        /*!
         *  * `const SizeType &... indices`: parameter pack, indices of the tensor
         *      * SizeType must be integer type e.g.: `int`, `size_t`,...
         *
         *  \n
         *  Behavior:\n
         *  each index in parameter pack `indices` corresponds to the index
         *
         * */
        template<typename ...SizeType>
        constexpr const Type & operator()(const SizeType &... indices) const {
            return this->data_.template data_ptr<Type>()[this->linearIndex_(indices...)];
        }

        //! Explicitly access the torch::Tensor
        /*! \todo: Add Documentation*/
        friend torch::Tensor to_torch(const Tensor<Type> & other) {
            return other.data_;
        }

        //! Access the underlying pointer (CPU only)
        constexpr Type * data(){
            return this->data_.template data_ptr<Type>();
        }

        //! Access the underlying pointer (CPU only)
        constexpr const Type * data() const {
            return this->data_.template data_ptr<Type>();
        }

        // =====================================================================
        // Slice Operation
        // =====================================================================

        //! Slice the Tensors `dim`th dimension from `start` to `end` with taking only every `step`th element.
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step = 1){
            torch::Tensor slice = data_.slice(dim,start,end,step);
            return Tensor<Type,RealType> (std::move(slice));
        }

        //! Slice the Tensors `dim`th dimension from `start` to `end` with taking only every `step`th element.
        /*! \todo: Add Documentation*/
        const Tensor<Type,RealType> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step = 1) const {
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return Tensor<Type,RealType>(std::move(slice));
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
        Tensor<bool> operator== (const NSL::Tensor<Type> & other) const {
            return Tensor<bool>(this->data_ == other.data_);
        }

        //! Elementwise equal: Tensor to number
        Tensor<bool> operator== (const Type & value) const {
            return Tensor<bool>(this->data_ == value);
        }

        //! Elementwise not equal: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator!= (const NSL::Tensor<Type,RealType> & other) const {
            return Tensor<bool>(this->data_ != other.data_);
        }

        //! Elementwise not equal: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator!= (const Type & value) const {
            return Tensor<bool>(this->data_ != value);
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator<= (const NSL::Tensor<Type,RealType> & other) const {
            return Tensor<bool>(this->data_ <= other.data_);
        }

        //! Elementwise smaller or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator<= (const Type & value) const {
            return Tensor<bool>(this->data_ <= value);
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator>= (const NSL::Tensor<Type,RealType> & other) const {
            return Tensor<bool>(this->data_ >= other.data_);
        }

        //! Elementwise greater or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator>= (const Type & value) const {
            return Tensor<bool>(this->data_ >= value);
        }

        //! Elementwise smaller : Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator< (const NSL::Tensor<Type,RealType> & other) const {
            return Tensor<bool>(this->data_ < other.data_);
        }

        //! Elementwise smaller: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator< (const Type & value) const {
            return Tensor<bool>(this->data_ < value);
        }

        //! Elementwise smaller or equals: Tensor to Tensor
        /*! \todo: Add Documentation*/
        Tensor<bool> operator> (const NSL::Tensor<Type,RealType> & other) const {
            return Tensor<bool>(this->data_ > other.data_);
        }

        //! Elementwise greater or equals: Tensor to number
        /*! \todo: Add Documentation*/
        Tensor<bool> operator> (const Type & value) const {
            return Tensor<bool>(this->data_ > value);
        }

        constexpr bool is_complex(){
            return std::is_same<Type,RealType>();
        }

        // =====================================================================
        // Print and stream
        // =====================================================================

        //! Streaming operator
        //! \todo: We should code up our own.
        friend std::ostream & operator << (std::ostream & os, const Tensor<Type> & T){
            return (os << T.data_);
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
         * to the constructor `NSL::Tensor<Type,RealType>::Tensor(Arg size0, SizeType... sizes)`.
         * */
        [[nodiscard]] size_t shape(const size_t & dim) const {
            return data_.size(dim);
        }

        //! Get the extents of the tensor
        [[nodiscard]] std::vector<size_t> shape() const {
            std::vector<size_t> out(data_.dim());
            torch::IntArrayRef shape = this->data_.sizes();
            std::copy(shape.begin(),shape.end(),out.begin());
            return out;
        }

        //! Get the dimension of the Tensor.
        /*!
         *  The dimension of the tensor is specified at construction by the number
         *  of integer arguments provided to the constructor `NSL::Tensor(Arg size0, SizeType... sizes)`
         * */
        [[nodiscard]] size_t dim() const {
            return this->data_.dim();
        }

        //! Get the total number of elements.
        /*! \todo: Add Documentaton*/
        [[nodiscard]] size_t numel() const {
            return this->data_.numel();
        }

        // =====================================================================
        // Determinant
        // =====================================================================

        // Other .member functions are mutations which change .data_.
        // Since .det and .logdet shouldn't do that, they go in LinAlg.

        // =====================================================================
        // Transpose + Adjoint
        // =====================================================================

        // TODO: transpose (and maybe adjoint) could be a view?

        //! Transpose dim0 dim1
        /*!
         * \todo Add documentation
         * */
        NSL::Tensor<Type> & transpose(const size_t dim0, const size_t dim1) {
            data_ = torch::transpose(data_, dim0, dim1);
            return *this;
        }

        //! Matrix transpose
        /*!
         * \todo: Add documentation
         * */
        NSL::Tensor<Type> & transpose() {
            this->transpose(this->dim()-1, this->dim()-2);
            return *this;
        }

        //! Adjoint (elementwise complex conjugate & transpose) of `dim0` and `dim1`
        /*!
         * \todo: Add documentation
         * */
        NSL::Tensor<Type> & adjoint(const size_t dim0, const size_t dim1) {
            data_ = torch::transpose(data_, dim0, dim1).conj();
            return *this;
        }

        //! Matrix adjoint (elementwise complex conjugate & matrix transpose)
        /*!
         * \todo: Add documentation
         * */
        NSL::Tensor<Type> & adjoint() {
            this->adjoint(this->dim()-1, this->dim()-2);
            return *this;
        }

        //! Complex Conjugation (Elementwise)
        /*!
         * \todo: Add documentation
         * */
        NSL::Tensor<Type> & conj() {
            if constexpr(NSL::is_complex<Type>()){
                this->data_ = this->data_.conj();
            }
            return *this;
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
        Tensor<Type,RealType> & operator+=(const Type & value){
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

        //! Elementwise multiplication (Schur,Hadamard product): Tensor(Complex) * Tensor(Real)
        /*!
         * \todo Add documentation.
         */
        template<typename OtherType, typename OtherRealType = typename NSL::RT_extractor<OtherType>::value_type>
        Tensor<NSL::complex<RealType>,RealType> operator-(const Tensor<OtherType,OtherRealType> & other){
            // check for same precision
            static_assert(std::is_same<OtherRealType,RealType>());
            // ensure that at least one is complex valued
            static_assert(NSL::is_complex<Type>() || NSL::is_complex<OtherType>());
            // We expect the case where Type = RealType or OtherType = Type is handled by the operator<Type,RealType>
            Tensor<NSL::complex<RealType>,RealType> tmp(this->data_ - to_torch(other));
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


        //! Elementwise multiplication (Schur,Hadamard product): Tensor(Complex) - number(Real)
        /*!
         * \todo Add documentation.
         */
        template<typename OtherType>
        Tensor<NSL::complex<RealType>,RealType> operator-(const OtherType & other){
            // check for same precision
            static_assert(std::is_same<typename RT_extractor<OtherType>::value_type ,RealType>());
            // ensure that at least one is complex valued
            static_assert(NSL::is_complex<Type>() || NSL::is_complex<OtherType>());
            // We expect the case where Type = RealType or OtherType = Type is handled by the operator<Type,RealType>
            Tensor<NSL::complex<RealType>,RealType> tmp(this->data_ - other);
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
        Tensor<Type,RealType> & operator-=(const Type & value){
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

        //! Elementwise multiplication (Schur,Hadamard product): Tensor(Complex) * Tensor(Real)
        /*!
         * \todo Add documentation.
         */
        template<typename OtherType, typename OtherRealType = typename NSL::RT_extractor<OtherType>::value_type>
        Tensor<NSL::complex<RealType>,RealType> operator*(const Tensor<OtherType,OtherRealType> & other){
            // check for same precision
            static_assert(std::is_same<OtherRealType,RealType>());
            // ensure that at least one is complex valued
            static_assert(NSL::is_complex<Type>() || NSL::is_complex<OtherType>());
            // We expect the case where Type = RealType or OtherType = Type is handled by the operator<Type,RealType>
            Tensor<NSL::complex<RealType>,RealType> tmp(this->data_ * to_torch(other));
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

        //! Elementwise multiplication (Schur,Hadamard product): Tensor(Complex) * number(Real)
        /*!
         * \todo Add documentation.
         */
        template<typename OtherType>
        Tensor<NSL::complex<RealType>,RealType> operator*(const OtherType & other){
            // check for same precision
            static_assert(std::is_same<typename RT_extractor<OtherType>::value_type ,RealType>());
            // ensure that at least one is complex valued
            static_assert(NSL::is_complex<Type>() || NSL::is_complex<OtherType>());
            // We expect the case where Type = RealType or OtherType = Type is handled by the operator<Type,RealType>
            Tensor<NSL::complex<RealType>,RealType> tmp(this->data_ * other);
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
        Tensor<Type,RealType> & operator*=(const Type & value){
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
        Tensor<Type,RealType> & operator/=(const Type & value){
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
            if constexpr(NSL::is_complex<Type>()){
                return Tensor<RealType>(torch::real(this->data_));
            } else {
                return *this;
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
            if constexpr(NSL::is_complex<Type>()){
                return Tensor<RealType>(torch::imag(this->data_));
            } else {
                return Tensor<RealType>(torch::zeros(this->data_.sizes(),
                                    torch::TensorOptions().dtype<RealType>()
                                            .layout(this->data_.layout())
                                            .device(this->data_.device())
                ));
            }
        }

        // =====================================================================
        // 2D Linear Algebra
        // =====================================================================

        //! Matrix exponential.
        /*!
         * \todo: Add documentation
         * */
        Tensor<Type> & mat_exp() {
            data_ = data_.matrix_exp();
            return *this;
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
        Tensor<Type,RealType> sum(const size_t dim){
            return Tensor<Type,RealType>(this->data_.sum(dim));
        }

        //! Reduction: +
        /*! \todo: Add Documentation*/
        Type sum(){
            return this->data_.sum().template item<Type>();
        }

        //! Reduction: *
        /*! \todo: Add Documentation*/
        Tensor<Type,RealType> prod(const size_t dim){
            return Tensor<Type,RealType>(this->data_.prod(dim));
        }

        //! Reduction: *
        /*! \todo: Add Documentation*/
        Type prod(){
            return this->data_.prod().template item<Type>();
        }

        //! Reduction: && (logical and)
        /*! \todo: Add Documentation*/
        Tensor<bool> all(const size_t dim){
            assert((std::is_same<Type,bool>()));
            return Tensor<bool>(this->data_.all(dim));
        }

        //! Reduction: && (logical and)
        /*! \todo: Add Documentation*/
        Type all(){
            assert((std::is_same<Type,bool>()));
            return this->data_.all().template item<Type>();
        }

        //! Reduction: || (logical or)
        /*! \todo: Add Documentation*/
        Tensor<bool> any(const size_t dim){
            assert((std::is_same<Type,bool>()));
            return Tensor<bool>(this->data_.any(dim));
        }

        //! Reduction: || (logical or)
        /*! \todo: Add Documentation*/
        Type any(){
            assert((std::is_same<Type,bool>()));
            return this->data_.any().template item<Type>();
        }


        // =====================================================================
        // Resize operations
        // =====================================================================

        //! Expanding the Tensor by one dimension with size `newSize`
        /*! \todo: Add Documentation
         * */
        NSL::Tensor<Type> & expand(const size_t & newSize) {
            std::vector<size_t> sizes = this->shape();
            sizes.push_back(newSize);
            this->data_ = data_.unsqueeze(-1).expand(torch::IntArrayRef(sizes.data(),sizes.size()));
            return *this;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const size_t & shift){
            this->data_ = this->data_.roll(shift,0);
            return *this;
        }

        //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const size_t & shift, const size_t & dim){
            this->data_ = this->data_.roll(shift,dim);
            return *this;
        }

        //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
        /*! \todo: Add Documentation*/
        NSL::Tensor<Type,RealType> & shift(const size_t & shift, const Type & boundary){
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
        NSL::Tensor<Type,RealType> & shift(const size_t & shift, const size_t & dim, const Type &boundary){
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
        template<typename... SizeType>
        inline std::size_t linearIndex_(const SizeType &... indices) const{
            // check that all arguments of the parameter pack are convertible to
            // the defined size type (i.e. integer valued)
            static_assert(NSL::all_convertible<size_t, SizeType...>::value);

            // check that the number of arguments in indices matches the dimension of the tensor
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const SizeType &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const SizeType &... indices) called with to many indices");

            // unpack the parameter pack
            std::array<size_t, sizeof...(indices)> a_indices = {indices...};

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
