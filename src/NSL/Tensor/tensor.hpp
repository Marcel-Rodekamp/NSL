#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

#include <torch/torch.h>
#include <vector>
#include <array>
#include <iomanip>
#include "../assert.hpp"
#include "../complex.hpp"

//! Imported Namespace: torch::indexing
using namespace torch::indexing;

namespace NSL {

// doxygen documentation
//! Representation of a multidimensional data.

//! Storing data in various data types is one of the key requirements of any simulation.
//! This class provides an interface to torch::Tensor (libtorch) in order to allow
//! access and functionality on various architectures such as Multi CPU setups, Nvidia GPUs, AMD GPUs, ...
template<typename Type>
class Tensor {

    //! Alias: size_t = long int
    using size_t = long int;


    private:
        // underlying data
        torch::Tensor data_;

        // transform given indices and strides in data_ to a linear index accessing the underlying pointer
        template<typename... Args>
        inline const std::size_t linearIndex_(const Args &... indices){
            std::array<long int, sizeof...(indices)> a_indices = {indices...};
            std::size_t offset = 0;

            for(long int d = 0 ; d < sizeof...(indices); ++d){
                offset += a_indices[d] * data_.stride(d);
            }

            return offset;
        }

    public:
        //Default constructor not required
        constexpr explicit Tensor() = delete;

        //! D-dimensional constructor.

        //! Constructs the Tensor with D dimensions. D is determined by the number of arguments provided.\n
        //! \n
        //! Params:\n
        //!     * `size0`: Extend of the 0th dimension\n
        //!     * `sizes`: Parameter pack, extends of respective dimensions
        //!
        //! \n
        //! Assumptions:\n
        //!     * At least one argument must be passed (`size0`)
        //!     * ArgsT must be of integral type (convertible to `size_t`)
        //!     * Tested Types: `bool`, `float`, `double`, `NSL::complex<float>`, `NSL::complex<double>`
        //!
        //!
        //! \n
        //! Further behavior:\n
        //!     * Initialization sets all values to `Type` equivalent of 0
        template<typename... ArgsT>
        constexpr explicit Tensor(const size_t & size0, const ArgsT &... sizes):
            data_(torch::zeros({size0, sizes...},torch::TensorOptions().dtype<Type>().device(torch::kCPU))) {}

        //Construct data_ with sizes stored in dims.
        [[deprecated("Will be deleted in version 1, use variadic constructors")]]
        constexpr explicit Tensor(const std::vector<long int>  & dims):
            data_(torch::zeros(dims, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        constexpr explicit Tensor(torch::Tensor && other):
            data_(std::move(other))
        {}

        constexpr explicit Tensor(torch::Tensor & other):
            data_(other)
        {}

        // copy constructor
        constexpr Tensor(const Tensor& other):
            data_(other.data_)
        {}

        // move constructor
        constexpr Tensor(Tensor && other) noexcept:
            data_(std::move(other.data_))
        {}

        //Fill with random values
        Tensor<Type> & rand(){
            this->data_ = torch::rand_like(this->data_);
            return (*this);
        }

        Tensor<Type> & copy(Tensor<Type> & other){
            data_ = other.data_.clone();
            return *this;
        }

        Tensor<Type> & copy(Tensor<Type> other){
            data_ = other.data_.clone();
            return *this;
        }


        // =====================================================================
        // Random Access Operators
        // =====================================================================
        template<typename ...Args>
        constexpr Type & operator()(const Args &... indices){
            static_assert(NSL::all_convertible<size_t, Args...>::value,
                    "NSL::Tensor::operator()(const Args &... indices) can only be called with arguments of integer type"
            ); // static_assert
            // need data_.dim() arguments!
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const Args &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const Args &... indices) called with to many indices");

            // ToDo: data_.dim() == 1 is a problem as the slice case would always be called! Unfortunately, data_.dim() is only known at runtime
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

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr Tensor<Type> operator[](const size_t index){
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr const Tensor<Type> & operator[](const size_t index) const {
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr Type & operator[](const std::initializer_list<size_t> & indices) {
            // get index transformation
            // compare TensorAccessor: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
            // last accessed 20.08.21
            size_t offset = 0;
            for(size_t d = 0; d < indices.size(); ++d){
                offset += *(indices.begin()+d) * data_.stride(d);
            }

            // dereference underlying pointer
            return data_.data_ptr<Type>()[offset];
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr Type & operator[](const std::initializer_list<size_t> & indices) const {
            // get index transformation
            // compare TensorAccessor: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
            // last accessed 20.08.21
            size_t offset = 0;
            for(size_t d = 0; d < indices.size(); ++d){
                offset += *(indices.begin()+d) * data_.stride(d);
            }

            // dereference underlying pointer
            return data_.data_ptr<Type>()[offset];
        }

        friend torch::Tensor to_torch(const Tensor<Type> & other) {
            return other.data_;
        }

        torch::Tensor to_torch(){
            return data_;
        }

        constexpr Type * data(){
            return data_.data_ptr<Type>();
        }

        constexpr Type * data() const {
            return data_.data_ptr<Type>();
        }

        Tensor<Type> real(){
            return NSL::Tensor<Type>(torch::real(data_));
        }

        Tensor<Type> real() const {
            return NSL::Tensor<Type>(torch::real(data_));
        }

        Tensor<Type> imag(){
            return NSL::Tensor<Type>(torch::imag(data_));
        }

        Tensor<Type> imag() const {
            return NSL::Tensor<Type>(torch::imag(data_));
        }

        // =====================================================================
        // Slice Operation
        // =====================================================================
        constexpr Tensor<Type> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step){
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return Tensor<Type>(slice);
        }

        constexpr Tensor<Type> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step) const {
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return Tensor<Type>(slice);
        }


        // =====================================================================
        // Algebra Operators
        // =====================================================================
        //Product by a scalar.
        Tensor<Type> operator*(const Type & factor){
            Tensor <Type> out (data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = factor* out.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        Tensor<Type> & prod(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = factor* this->data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        Tensor<Type> & operator*=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = factor* this->data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Product by a Tensor
        Tensor<Type> operator*(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::Tensor <Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = out.data_.template data_ptr<Type>()[x]*other.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        Tensor<Type> & operator*=(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]* other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        Tensor<Type> & prod(const Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]* other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //sum by Tensor
        Tensor<Type> operator+(Tensor<Type> & other ) {
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::Tensor out(this->data_);
            for (long int x = 0; x < out.data_.numel(); ++x) {
                out.data_.template data_ptr<Type>()[x] =
                        out.data_.template data_ptr<Type>()[x] + other.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        Tensor<Type> & operator+=(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        Tensor<Type> & sum(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Sum by a scalar.
        Tensor<Type> operator+(const Type & factor){
            NSL::Tensor <Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = factor + out.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        Tensor<Type> & operator+=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+factor;
            }
            return *this;
        }

        Tensor<Type> & sum(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+factor;
            }
            return *this;
        }

        //Minus by Tensor
        Tensor<Type> operator-(Tensor<Type> & other ) {
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::Tensor out(this->data_);
            for (long int x = 0; x < out.data_.numel(); ++x) {
                out.data_.template data_ptr<Type>()[x] =
                        out.data_.template data_ptr<Type>()[x] - other.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        Tensor<Type> & operator-=(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        Tensor<Type> & subs(const NSL::Tensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Minus by a scalar.
        Tensor<Type> operator-(const Type & factor){
            NSL::Tensor <Type> out (this->data_);
            for(long int x = 0; x < data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = out.data_.template data_ptr<Type>()[x]-factor;
            }
            return out;
        }

        Tensor<Type> & operator-=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-factor;
            }
            return *this;
        }

        Tensor<Type> & subs(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-factor;
            }
            return *this;
        }

        //Equal
        Tensor<Type> & operator=(const Tensor<Type> & other){
            assert(other.data_.dim() == this->data_.dim());

            // copy the data explicitly by looping over all elements
            for(long int x = 0; x < other.data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = other.data_.template data_ptr<Type>()[x];
            }

            return *this;
        }
        // =====================================================================
        // Boolean operators
        // =====================================================================

        //Comparison each element of two tensors.
        bool operator== (NSL::Tensor<Type> & other) const{
            assert(other.data_.sizes() == this->data_.sizes());
            bool out = true;
            for(long int x = 0; x < other.data_.numel(); ++x) {
                if (this->data_.template data_ptr<Type>()[x] != other.data_.template data_ptr<Type>()[x]){
                    out = false;
                    break;
                }
            }
            return out;
        }

        //comparison all element with a number.
        bool operator== (const Type & num) const{
            bool out = true;
            for(long int x = 0; x < this->data_.numel(); ++x) {
                if (this->data_.template data_ptr<Type>()[x] != num){
                    out = false;
                    break;
                }
            }
            return out;
        }

        // =====================================================================
        // Print and stream
        // =====================================================================

        // streaming operator
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

        //ToDo: Revise static_cast.
        //Return size of dimension in data_.
        [[nodiscard]] std::size_t shape(const std::size_t & dim) const {
            return data_.size(static_cast<const long int>(dim));
        }

        [[nodiscard]] std::vector<long int> shape() const {
            std::vector<long int> out(data_.dim());
            for (long int i=0; i< data_.dim(); ++i){
                out[i] = data_.size(i);
            }
            return out;
        }

        [[nodiscard]] std::size_t dim() const {
            return this->data_.dim();
        }

        // =====================================================================
        // Exponential
        // =====================================================================

        //Tensor exponential.
        Tensor<Type> & exp() {
            data_.exp_();
            return *this;
        }

        //Matrix exponential.
        Tensor<Type> & mat_exp() {
            data_ = data_.matrix_exp();
            return *this;
        }

        // =====================================================================
        // Determinant
        // =====================================================================

        // TODO: it seems to me that .det and .logdet should not be mutations...
        // nor should computing it require copying.  But for consistency,
        // for now, they are mutations.

        NSL::Tensor<Type> & det() {
            data_ = data_.det();
            return *this;
        }

        NSL::Tensor<Type> logdet() {
            data_ = data_.logdet();
            return *this;
        }

        // =====================================================================
        // Transpose + Adjoint
        // =====================================================================

        // TODO: transpose (and maybe adjoint) could be a view?

        NSL::Tensor<Type> & transpose(const size_t dim0, const size_t dim1) {
            data_ = torch::transpose(data_, dim0, dim1);
            return *this;
        }

        NSL::Tensor<Type> & transpose() {
            this->transpose(this->dim()-1, this->dim()-2);
            return *this;
        }

        NSL::Tensor<Type> & adjoint(const size_t dim0, const size_t dim1) {
            data_ = torch::transpose(data_, dim0, dim1).conj();
            return *this;
        }

        NSL::Tensor<Type> & adjoint() {
            this->adjoint(this->dim()-1, this->dim()-2);
            return *this;
        }

        // =====================================================================
        // Expand
        // =====================================================================
        template<typename... Args>
        NSL::Tensor<Type> & expand(const Args &... dims) {
            this->data_ = data_.unsqueeze(-1).expand({dims...});
            return *this;
        }

        //Tensor Expand
        [[deprecated("Will be deleted in version 1, use expand(Args... & dims)")]]
        NSL::Tensor<Type> & expand(std::deque<long int> dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        //Tensor Expand.
        [[deprecated("Will be deleted in version 1, use expand(Args... & dims)")]]
        NSL::Tensor<Type> & expand(const size_t & dimension){
            std::deque<long int> dims;
            dims.push_front(dimension);
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        // =====================================================================
        // Shift
        // =====================================================================

        // return the t-shifted version of this tensor.
        //Why use roll? https://discuss.pytorch.org/t/is-there-a-way-to-create-a-rolled-shifted-and-wrapped-view-of-a-tensor/67568
        NSL::Tensor<Type> & shift(const long int & shift, const Type & boundary){
            data_ = torch::roll(data_,shift,0);
            std::cout<<"Boundary:" <<boundary<<std::endl;

            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                    for(long int x = 0; x < this->data_[i].numel(); ++x){
                        this->data_[i].template data_ptr<Type>()[x] = boundary*data_[i].template data_ptr<Type>()[x];
                    }
                }
            }
            else{
                for (int i = (data_).dim() + shift; i < (data_).dim(); i++){
                    for(long int x = 0; x < this->data_[i].numel(); ++x){
                        this->data_[i].template data_ptr<Type>()[x] = boundary*data_[i].template data_ptr<Type>()[x];
                    }
                }
            }
            return *this;
        }

        NSL::Tensor<Type> & shift(const long int & shift) {
            data_ = torch::roll(data_, shift, 0);
            return *this;
        }
    }; //Tensor class


//========================================================================
//Time Tensor Class

template<typename Type>
class TimeTensor {

    // alias settings
    using size_t = long int;


    private:
        torch::Tensor data_;

        template<typename... Args>
        inline const std::size_t linearIndex_(const Args &... indices){
            std::array<long int, sizeof...(indices)> a_indices = {indices...};
            std::size_t offset = 0;

            for(long int d = 0 ; d < sizeof...(indices); ++d){
                offset += a_indices[d] * data_.stride(d);
            }

            return offset;
        }

    public:
        //Default constructor.
        constexpr explicit TimeTensor() = default;

        //Construct with N dimensions
        template<typename... ArgsT>
        constexpr explicit TimeTensor(const size_t & size0, const ArgsT &... sizes):
            data_(torch::zeros({size0, sizes...},torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        //Construct data_ with sizes stored in dims.
        [[deprecated("Will be deleted in version 1, use variadic constructors")]]
        constexpr explicit TimeTensor(const std::vector<long int>  & dims):
            data_(torch::zeros(dims, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        constexpr explicit TimeTensor(torch::Tensor && other):
            data_(std::move(other))
        {}

        constexpr explicit TimeTensor(torch::Tensor & other):
            data_(other)
        {}

        // copy constructor
        constexpr TimeTensor(const TimeTensor& other):
            data_(other.data_)
        {}

        // move constructor
        constexpr TimeTensor(TimeTensor && other) noexcept:
            data_(std::move(other.data_))
        {}

        //Random creation
        TimeTensor<Type> & rand(){
            data_=torch::rand(data_.sizes(), torch::TensorOptions().dtype<Type>().device(torch::kCPU));
            return (*this);
        }

        // ToDo: Repalace these calls with the assignment/copy constructor
        TimeTensor<Type> & copy(TimeTensor<Type> & other){
            data_ = other.data_.clone();
            return *this;
        }


        // =====================================================================
        // Random Access Operators
        // =====================================================================
        template<typename ...Args>
        constexpr Type & operator()(const Args &... indices){
            static_assert(NSL::all_convertible<size_t, Args...>::value,
                          "NSL::TimeTensor::operator()(const Args &... indices) can only be called with arguments of integer type"
            ); // static_assert
            // need data_.dim() arguments!
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const Args &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const Args &... indices) called with to many indices");

            // ToDo: data_.dim() == 1 is a problem as the slice case would always be called! Unfortunately, data_.dim() is only known at runtime
            return data_.data_ptr<Type>()[linearIndex_(indices...)];
        }

        template<typename ...Args>
        constexpr Type & operator()(const Args &... indices) const {
            static_assert(NSL::all_convertible<size_t, Args...>::value,
                      "NSL::TimeTensor::operator()(const Args &... indices) can only be called with arguments of integer type"
            ); // static_assert
            // need data_.dim() arguments!
            assertm(!(sizeof...(indices) < data_.dim()), "operator()(const Args &... indices) called with to little indices");
            assertm(!(sizeof...(indices) > data_.dim()), "operator()(const Args &... indices) called with to many indices");

            // ToDo: data_.dim() == 1 is a problem as the slice case would always be called! Unfortunately, data_.dim() is only known at runtime
            return data_.data_ptr<Type>()[linearIndex_(indices...)];
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr Tensor<Type> operator[](const size_t index){
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr const Tensor<Type> & operator[](const size_t index) const {
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr Type & operator[](const std::initializer_list<size_t> & indices) {
            // get index transformation
            // compare TensorAccessor: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
            // last accessed 20.08.21
            size_t offset = 0;
            for(size_t d = 0; d < indices.size(); ++d){
                offset += *(indices.begin()+d) * data_.stride(d);
            }

            // dereference underlying pointer
            return data_.data_ptr<Type>()[offset];
        }

        [[deprecated("Will be deleted in version 1, use operator()")]]
        constexpr const Type & operator[](const std::initializer_list<size_t> & indices) const {
            // get index transformation
            // compare TensorAccessor: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h
            // last accessed 20.08.21
            size_t offset = 0;
            for(size_t d = 0; d < indices.size(); ++d){
                offset += *(indices.begin()+d) * data_.stride(d);
            }

            // dereference underlying pointer
            return data_.data_ptr<Type>()[offset];
        }

        friend torch::Tensor to_torch(const TimeTensor<Type> & other){
            return other.data_;
        }

        torch::Tensor & to_torch(){
            return data_;
        }

        constexpr Type * data(){
            return data_.data_ptr<Type>();
        }

        constexpr Type * data() const {
            return data_.data_ptr<Type>();
        }

        TimeTensor<Type> real(){
            return NSL::TimeTensor<Type>(torch::real(data_));
        }

        TimeTensor<Type> real() const {
            return NSL::TimeTensor<Type>(torch::real(data_));
        }

        TimeTensor<Type> imag(){
            return NSL::TimeTensor<Type>(torch::imag(data_));
        }

        TimeTensor<Type> imag() const {
            return NSL::TimeTensor<Type>(torch::imag(data_));
        }

    // =====================================================================
        // Slice Operation
        // =====================================================================
        constexpr TimeTensor<Type> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step){
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return TimeTensor<Type>(slice);
        }

        constexpr TimeTensor<Type> slice(const size_t & dim, const size_t & start, const size_t & end , const size_t & step) const {
            torch::Tensor && slice = data_.slice(dim,start,end,step);
            return TimeTensor<Type>(slice);
        }

        // =====================================================================
        // Algebra Operators
        // =====================================================================

        // ToDo: merge with prod
        //Product by a scalar.
        TimeTensor<Type> operator*(const Type & factor) const{
            TimeTensor <Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = factor* out.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        TimeTensor<Type> & prod(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = factor* this->data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        TimeTensor<Type> & operator*=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = factor* this->data_.template data_ptr<Type>()[x];
            }
            return *this;
        }


        //Product by a Timetensor
        TimeTensor<Type> operator*(const TimeTensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::TimeTensor <Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]*other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        TimeTensor<Type> & operator*=(const TimeTensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]* other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        TimeTensor<Type> & prod(const TimeTensor<Type> & other){
            assert(other.data_.sizes() == this->data_.sizes());
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]* other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Sum of TimeTensors
        TimeTensor<Type> operator+(TimeTensor<Type> & other ){
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::TimeTensor<Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = out.data_.template data_ptr<Type>()[x] + other.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        TimeTensor<Type> & operator+=(TimeTensor<Type> & other ){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x] + other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        TimeTensor<Type> & sum(TimeTensor<Type> & other ){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x] + other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Sum by a scalar.
        TimeTensor<Type> operator+(const Type & factor) const{
            NSL::TimeTensor<Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = factor + out.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        TimeTensor<Type> & operator+=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+factor;
            }
            return *this;
        }

        TimeTensor<Type> & sum(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]+factor;
            }
            return *this;
        }

        //Minus of TimeTensors
        TimeTensor<Type> operator-(TimeTensor<Type> & other ){
            assert(other.data_.sizes() == this->data_.sizes());
            NSL::TimeTensor<Type> out (this->data_);
            for(long int x = 0; x < out.data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = out.data_.template data_ptr<Type>()[x] - other.data_.template data_ptr<Type>()[x];
            }
            return out;
        }

        TimeTensor<Type> & operator-=(TimeTensor<Type> & other ){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x] - other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        TimeTensor<Type> & subs(TimeTensor<Type> & other ){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x] - other.data_.template data_ptr<Type>()[x];
            }
            return *this;
        }

        //Minus by a scalar.
        TimeTensor<Type> operator-(const Type & factor){
            TimeTensor <Type> out (this->data_);
            for(long int x = 0; x < data_.numel(); ++x){
                out.data_.template data_ptr<Type>()[x] = out.data_.template data_ptr<Type>()[x]-factor;
            }
            return out;
        }

        TimeTensor<Type> & operator-=(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-factor;
            }
            return *this;
        }

        TimeTensor<Type> & subs(const Type & factor){
            for(long int x = 0; x < this->data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = this->data_.template data_ptr<Type>()[x]-factor;
            }
            return *this;
        }

        //Equal
        TimeTensor<Type> & operator=(const TimeTensor<Type> & other){
            assert(other.data_.dim() == this->data_.dim());

            // copy the data explicitly by looping over all elements
            for(long int x = 0; x < other.data_.numel(); ++x){
                this->data_.template data_ptr<Type>()[x] = other.data_.template data_ptr<Type>()[x];
            }

            return *this;
        }


        // =====================================================================
        // Boolean operators
        // =====================================================================
        //Comparison each element of two tensors.
        bool operator== (NSL::TimeTensor<Type> & other) const{
            assert(other.data_.sizes() == this->data_.sizes());
            bool out = true;
            for(long int x = 0; x < other.data_.numel(); ++x) {
                if (this->data_.template data_ptr<Type>()[x] != other.data_.template data_ptr<Type>()[x]){
                    out = false;
                    break;
                }
            }
            return out;
        }
        //comparison all element with a number.
        bool operator== (const Type & num) const{
            bool out = true;
            for(long int x = 0; x < this->data_.numel(); ++x) {
                if (this->data_.template data_ptr<Type>()[x] != num){
                    out = false;
                    break;
                }
            }
            return out;
        }

        // =====================================================================
        // Print and stream
        // =====================================================================

        // streaming operator
        friend std::ostream & operator << (std::ostream & os, const TimeTensor<Type> & T){
            return (os << T.data_);
        }

        //Print tensor.
        void print (){
            std::cout<<data_ <<std::endl;
        }

        //Print tensor complex.
        void print_complex (){
            std::cout<< torch::view_as_real(data_)<<std::endl;
        }


        // =====================================================================
        // Shape and dimension
        // =====================================================================

        //Return size of dimension in data_.
        //ToDo: Revise static_cast.
        //Return size of dimension in data_.
        [[nodiscard]] std::size_t shape(const size_t & dim) const {
            return data_.size(static_cast<const long int>(dim));
        }

        [[nodiscard]] std::vector<long int> shape() const {
            std::vector<long int> out;
            for (long int i=0; i< data_.dim(); ++i){
                out.push_back(data_.size(i));
            }
            return out;
        }

        [[nodiscard]] size_t dim() const {
            return this->data_.dim();
        }


        // =====================================================================
        // Exponential
        // =====================================================================

        //Tensor exponential.
        TimeTensor<Type> & exp() {
            data_.exp_();
            return *this;
        }

        // =====================================================================
        // Expand
        // =====================================================================

        template<typename... Args>
        NSL::TimeTensor<Type> & expand(const Args &... dims) {
            this->data_ = data_.unsqueeze(-1).expand({dims...});
            return *this;
        }

        //Tensor Expand.
        [[deprecated("Will be deleted in version 1, use expand(Args... & dims)")]]
        TimeTensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        [[deprecated("Will be deleted in version 1, use expand(Args... & dims)")]]
        TimeTensor<Type> & expand(std::deque<long int> dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        //Tensor Expand.
        [[deprecated("Will be deleted in version 1, use expand(Args... & dims)")]]
        TimeTensor<Type> & expand(const size_t & dimension){
            std::deque<long int> dims;
            dims.push_front(dimension);
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }


        // =====================================================================
        // Shift
        // =====================================================================

        // return the t-shifted version of this tensor.
        //Why use roll? https://discuss.pytorch.org/t/is-there-a-way-to-create-a-rolled-shifted-and-wrapped-view-of-a-tensor/67568
        NSL::TimeTensor<Type> & shift(const long int & shift, const Type & boundary){
            data_ = torch::roll(data_,shift,0);
            std::cout<<"Boundary:" <<boundary<<std::endl;
            if(shift>0) {
                for (int i = 0; i <  shift ; i++ ) {
                    for(long int x = 0; x < this->data_[i].numel(); ++x){
                        this->data_[i].template data_ptr<Type>()[x] = boundary*data_[i].template data_ptr<Type>()[x];
                    }
                }
            }
            else{
                for (int i = (data_).dim() + shift; i < (data_).dim(); i++){
                    for(long int x = 0; x < this->data_[i].numel(); ++x){
                        this->data_[i].template data_ptr<Type>()[x] = boundary*data_[i].template data_ptr<Type>()[x];
                    }
                }
            }
            return *this;
        }

        NSL::TimeTensor<Type> & shift(const long int & shift) {
            data_ = torch::roll(data_, shift, 0);
            return *this;
        }
    }; //Time Tensor class
} // namespace NSL

#endif //NSL_TENSOR_HPP
