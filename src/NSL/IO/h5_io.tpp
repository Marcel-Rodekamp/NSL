#ifndef NSL_IO_H5_IO_TPP
#define NSL_IO_H5_IO_TPP
#include "../complex.hpp"
#include "../concepts.hpp"
#include <iostream>
#include "highfive/H5File.hpp"

using namespace HighFive;

namespace NSL {

class H5IO {
public:
  H5IO(std::string h5file) :
    h5file_(h5file),
    h5f_(h5file, File::ReadWrite | File::OpenOrCreate | File::Truncate)
  {}

    template <NSL::Concept::isNumber Type> inline int write(const NSL::Tensor<Type> &tensor, const std::string node){
    
    std::string DIM("shape");
    std::vector<NSL::size_t> shape = tensor.shape();
    
    //    auto flat_tensor = tensor.flatten();
    if constexpr (NSL::is_complex<Type>()) {
      std::vector<std::complex<NSL::RealTypeOf<Type>>> phi(tensor.data(), tensor.data()+tensor.numel());

      DataSet dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(node,  DataSpace::From(phi));
      dataset.write(phi);
      
      // now write out the dimension of the tensor as an attribute
      Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(shape));
      dim.write(shape);
    } else {
      std::vector<Type> phi(tensor.data(), tensor.data()+tensor.numel());

      DataSet dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(node,  DataSpace::From(phi));
      dataset.write(phi);

      // now write out the dimension of the tensor as an attribute
      Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(shape));
      dim.write(shape);
    }
    
    return 0; 
  }

  //  template <NSL::Concept::isNumber Type> inline NSL::Tensor<Type> read(const std::string node){
    
  //    if(h5f_.exist(node)){ // check if the node exists
  //      return 0;
  //    } else {
      // node does not exist
  //      return 1;
  //    }
  //}
  
private:
  
  std::string h5file_;
  File h5f_;
  
    
};



} // namespace NSL

#endif // NSL_IO_H5_IO_TPP
