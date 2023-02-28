#ifndef NSL_IO_H5_IO_TPP
#define NSL_IO_H5_IO_TPP
#include "../complex.hpp"
#include "../concepts.hpp"
#include <iostream>
#include <vector>
#include "highfive/H5File.hpp"

using namespace HighFive;

namespace NSL {

class H5IO {
public:
  H5IO(std::string h5file) :
    h5file_(h5file),
    h5f_(h5file, File::ReadWrite | File::OpenOrCreate )
    {}
    
    H5IO(std::string h5file, auto FileHandle) :
    h5file_(h5file),
    h5f_(h5file, FileHandle)  //| File::Truncate)
    {}

    template <NSL::Concept::isNumber Type> inline int write(const NSL::Tensor<Type> &tensor, const std::string node){
    
	std::string DIM("shape");
    	std::vector<NSL::size_t> shape = tensor.shape();
    
	if constexpr (NSL::is_complex<Type>()) {
      	   std::vector<std::complex<NSL::RealTypeOf<Type>>> phi(tensor.data(), tensor.data()+tensor.numel());

      	   DataSet dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(node,  DataSpace::From(phi));
      	   dataset.write(phi);
      
	   // now write out the dimension of the tensor as an attribute
      	   Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(shape));
      	   dim.write(shape);
    	} else {
      	   std::vector<Type> phi(tensor.data(), tensor.data()+tensor.numel());

      	   DataSet dataset = h5f_.createDataSet<Type>(node,  DataSpace::From(phi));
      	   dataset.write(phi);

      	   // now write out the dimension of the tensor as an attribute
      	   Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(shape));
      	   dim.write(shape);
    	}
    
	return 0; 
    }

    template <NSL::Concept::isNumber Type> inline int write(const NSL::Configuration<Type> &config, const std::string node){
    
	for (auto [key,field] : config) {
	    if (node.back() == '/' ) {
	       this -> write(field, node+key);
	    } else {
	       this -> write(field, node+"/"+key);
	    }
	}
    
	return 0; 
    }

    template <NSL::Concept::isNumber Type> inline int read(NSL::Tensor<Type> &tensor,const std::string node){
    
      if(h5f_.exist(node)){ // check if the node exists
        DataSet dataset = h5f_.getDataSet(node);
	
	/* can use the following to list all attributes
      	std::vector<std::string> all_attributes_keys = dataset.listAttributeNames();
        for (const auto& attr: all_attributes_keys) {
            std::cout << "attribute: " << attr << std::endl;
        }
	*/
	
      	// first read the attributes to get the shape of the tensor
	std::vector<NSL::size_t> shape;
	Attribute dim = dataset.getAttribute("shape");
	dim.read(shape);
	int numElems = 1;

	for (NSL::size_t i=0;i<shape.size();i++)
	{ numElems *= shape[i]; }
	
	if(!tensor.defined()){
	   tensor = NSL::Tensor<Type> (numElems);
	} else {
	   tensor.flatten();  // flatten the array
	   if(tensor.numel() != numElems)
	   {
	      tensor.resize(numElems);  // Need to resize the array here!
	   }
	}

	// now get the data
	if constexpr (NSL::is_complex<Type>()) {
	   std::vector<std::complex<NSL::RealTypeOf<Type>>> phi;
	   dataset.read(phi);
	   tensor = phi;
	   tensor.reshape(shape);
	} else {
	   std::vector<Type> phi;
	   dataset.read(phi);
	   tensor = phi;
	   tensor.reshape(shape);
	}

        return 0;
      } else {
        // node does not exist
        std::cout << "# Error! Node " + node + " doesn't exist!" << std::endl;
        return 1;
      }

      // I assume that once phi is 'off the stack', its destructor will be called and its memory released
    }

    template <NSL::Concept::isNumber Type> inline int read(NSL::Configuration<Type> &config, const std::string node){
    
	for (auto & [key,field] : config) {
	    if (node.back() == '/' ){
	      this -> read(field, node+key);
	    } else {
	      this -> read(field, node+"/"+key);
	    }
	}
    
	return 0; 
    }
  
private:
  
  std::string h5file_;
  File h5f_;
  
    
};



} // namespace NSL

#endif // NSL_IO_H5_IO_TPP
