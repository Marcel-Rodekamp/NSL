#ifndef NANOSYSTEMLIBRARY_MAT_VEC_HPP
#define NANOSYSTEMLIBRARY_MAT_VEC_HPP
namespace NSL{
namespace LinAlg {

template<typename Type>
NSL::Tensor<Type> mat_vec(NSL::Tensor<Type> & matrix, NSL::Tensor<Type> & vector){
    //    matrix.get_underlying().mv(vector.get_underlying());
}

} // namespace LinAlg
} // namespace NSL
#endif //NANOSYSTEMLIBRARY_MAT_VEC_HPP
