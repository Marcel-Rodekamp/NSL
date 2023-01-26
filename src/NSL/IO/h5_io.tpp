#ifndef NSL_IO_H5_IO_TPP
#define NSL_IO_H5_IO_TPP
#include "../complex.hpp"
#include "../concepts.hpp"
#include <iostream>
#include "highfive/H5File.hpp"

using namespace HighFive;

namespace NSL {
template <NSL::Concept::isNumber Type>
inline int h5_write(const Type &tensor, const std::string node, const std::string h5file){

       File h5(h5file, File::ReadWrite | File::OpenOrCreate); // if h5file doesn't exist, it will be created
       auto flat_tensor = tensor.flatten();
       if constexpr (NSL::is_complex<Type>()) {
       	  std::vector<std::complex<NSL::RealTypeOf<Type>>> phi(flat_tensor.data(), flat_tensor.data()+flat_tensor.numel());
    	  h5.createDataSet(node, phi);
  } else {
    std::vector<Type> phi(flat_tensor.data(), flat_tensor.data()+flat_tensor.numel());
    h5.createDataSet(node, phi);
  }

       return 0;       
}

template <NSL::Concept::isNumber Type>
inline int h5_read(const Type &tensor, const std::string node, const std::string h5file){

       if(std::filesystem::exists(h5file)){
	File h5(h5file, File::Read | File::Open);
	return 0;
       } else {
       	 // file does not exist
       	 return 1;
       }
}

\*
inline std::string to_string(const Type &z){
    if constexpr(NSL::is_complex<Type>()){
        auto re = NSL::real(z);
        auto im = NSL::imag(z);
        if (im < 0){
            return std::to_string(re)+std::to_string(im)+"i";
        }
        return std::to_string(re)+"+"+std::to_string(im)+"i";
    } else {
        return std::to_string(z); 
    }
}
*/
} // namespace NSL

#endif // NSL_IO_H5_IO_TPP
