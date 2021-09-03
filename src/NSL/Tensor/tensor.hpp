#ifndef NSL_TENSOR_HPP
#define NSL_TENSOR_HPP

#include<torch/torch.h>
#include <memory>
#include<vector>
#include <iomanip>
#include <complex>
#include <cmath>
#include <utility>
#include "../assert.hpp"
#include "../complex.hpp"

using namespace torch::indexing;
// ============================================================================
// CPU Implementations
namespace NSL {
    template<typename Type>
    class Tensor {
        using size_t = long int;
        using complex = c10::complex<Type>;

    private:
        torch::Tensor data_;
    public:


        //Default constructor.

        constexpr explicit Tensor() = default;

        //Construct 1D tensor (i.e. array).
        constexpr explicit Tensor(const std::size_t & size):
        data_(torch::zeros({static_cast<const long int>(size)}, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        //Construct data_ with sizes stored in dims.
        constexpr explicit Tensor(const std::vector<long int>  & dims):
        data_(torch::zeros(dims, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        constexpr explicit Tensor(torch::Tensor && other):
        data_(other)
        {}

        constexpr explicit Tensor(torch::Tensor & other):
        data_(other)
        {}

        // copy constructor
        constexpr Tensor(const Tensor& other):
        data_(other.data_)
        {}

        //Random creation
        Tensor<Type> & rand(){
            data_=torch::rand(data_.sizes(), torch::TensorOptions().dtype<Type>().device(torch::kCPU));
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

        constexpr Tensor<Type> operator[](const size_t index){
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        constexpr const Tensor<Type> & operator[](const size_t index) const {
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

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

        friend const torch::Tensor to_torch(const Tensor<Type> & other){
            return other.data_;
        }

        torch::Tensor to_torch(){
            return data_;
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
        const std::size_t shape(const std::size_t & dim) const {
            return data_.size(static_cast<const long int>(dim));
        }

        std::vector<long int> shape() const {
            std::vector<long int> out;
            for (long int i=0; i< data_.dim(); ++i){
                out.push_back(data_.size(i));
            }
            return out;
        }

        const std::size_t dim() const {
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

        // =====================================================================
        // Expand
        // =====================================================================

        //ToDo: Have to be const?
        //Tensor Expand.
        NSL::Tensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        //ToDo: Have to be const?
        //Tensor Expand.
        NSL::Tensor<Type> & expand(std::deque<long int> dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        //Tensor Expand.
        NSL::Tensor<Type> & expand(const std::size_t & dimension){
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
        using size_t = long int;

    private:
        torch::Tensor data_;
    public:
        // torch::Tensor data_; //just to prove cout's in the main program.

        //Default constructor.
        constexpr explicit TimeTensor() = default;

        //Construct 1D tensor (i.e. array).
        constexpr explicit TimeTensor(const std::size_t & size):
        data_(torch::zeros({static_cast<const long int>(size)}, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        //Construct data_ with sizes stored in dims.
        constexpr explicit TimeTensor(const std::vector<long int>  & dims):
        data_(torch::zeros(dims, torch::TensorOptions().dtype<Type>().device(torch::kCPU)))
        {}

        constexpr explicit TimeTensor(torch::Tensor && other):
        data_(other)
        {}

        constexpr explicit TimeTensor(torch::Tensor & other):
        data_(other)
        {}

        // copy constructor
        constexpr TimeTensor(const TimeTensor& other):
        data_(other.data_)
        {}

        //Random creation
        TimeTensor<Type> & rand(){
            data_=torch::rand(data_.sizes(), torch::TensorOptions().dtype<Type>().device(torch::kCPU));
            return (*this);
        }

        TimeTensor<Type> & copy(TimeTensor<Type> & other){
            data_ = other.data_.clone();
            return *this;
        }
        // =====================================================================
        // Random Access Operators
        // =====================================================================

        constexpr Tensor<Type> operator[](const size_t index){
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

        constexpr const Tensor<Type> & operator[](const size_t index) const {
            torch::Tensor && slice = data_.slice(0,index,index+1,1).squeeze(0);
            return Tensor<Type>(slice);
        }

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

        friend const torch::Tensor to_torch(const TimeTensor<Type> & other){
            return other.data_;
        }

        torch::Tensor & to_torch(){
            return data_;
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
        const std::size_t shape(const std::size_t & dim) const {
            return data_.size(static_cast<const long int>(dim));
        }

        std::vector<long int> shape() const {
            std::vector<long int> out;
            for (long int i=0; i< data_.dim(); ++i){
                out.push_back(data_.size(i));
            }
            return out;
        }

        const std::size_t dim() const {
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

        //Tensor Expand.
       TimeTensor<Type> & expand(std::deque<long int> & dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        TimeTensor<Type> & expand(std::deque<long int> dims){
            std::for_each(data_.sizes().rbegin(), data_.sizes().rend(), [& dims](const long int & x){dims.push_front(x);});
            data_=(data_).unsqueeze(-1).expand(std::vector<long int>({dims.begin(), dims.end()})).clone();
            return *this;
        }

        //Tensor Expand.
        TimeTensor<Type> & expand(const std::size_t & dimension){
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
