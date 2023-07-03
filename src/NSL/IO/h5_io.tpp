#ifndef NSL_IO_H5_IO_TPP
#define NSL_IO_H5_IO_TPP

#include "../complex.hpp"
#include "../concepts.hpp"
#include "../logger.hpp"
#include <iostream>
#include <vector>
#include "device.tpp"
#include "highfive/H5File.hpp"
#include "MarkovChain/markovState.tpp"

namespace NSL {

using HighFive::File;

class H5IO {
    public:
        H5IO(std::string h5file) :
            h5file_(h5file),
            h5f_(h5file, NSL::File::ReadWrite | NSL::File::OpenOrCreate )
        {}
    
        H5IO(std::string h5file, auto FileHandle) :
            h5file_(h5file),
            h5f_(h5file, FileHandle)  //| File::Truncate)
        {}

        H5IO() : 
            h5file_("data.h5"),
            h5f_("data.h5", NSL::File::ReadWrite)
        {}

        HighFive::File &getFile() { // !!!!! WARNING !!!! FOR EXPERT USE ONLY !!!!! (KEEP AWAY FROM TOM!!!!) 
            return h5f_;
        }

        std::tuple<NSL::size_t,NSL::size_t> getMinMaxConfigs(std::string node) {
    	    NSL::size_t minCfg = 10000000000;
	        NSL::size_t maxCfg = -1;
	        NSL::size_t temp;

	        auto configs = h5f_.getGroup(node).listObjectNames();  // this list all the stored configuration numbers
            
            // this list is not given in ascending order  Really annoying!  I have to loop over them to find the most recent config. . .
            minCfg = std::stoi(configs[0]);
            for (int i=1;i<configs.size();i++){
	            temp = std::stoi(configs[i]);
                if (temp>maxCfg) {
	                maxCfg = temp;
                }
	            if(temp<minCfg && temp>0) {
	                minCfg = temp;
	            }
            }

            return {minCfg, maxCfg};
        } // getMinMaxConfigs

        template <NSL::Concept::isNumber Type> 
        inline int write(const NSL::MCMC::MarkovState<Type> & markovstate, const std::string node){
            std::string baseNode = node;
	
            // write out the configuration
    	    this -> write(markovstate.configuration, baseNode);

	        if constexpr (NSL::is_complex<Type>()) {
	            // write out the actionValue
	            HighFive::DataSet dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(
                    baseNode+"/actVal", 
                    HighFive::DataSpace::From(
                        static_cast<std::complex<NSL::RealTypeOf<Type>>>(markovstate.actionValue)
                    )
                );

	            dataset.write(static_cast <std::complex<NSL::RealTypeOf<Type>>> (markovstate.actionValue));

	            // write out the acceptanceProbability
	            dataset = h5f_.createDataSet<double>(
                    baseNode+"/acceptanceProbability",
                    HighFive::DataSpace::From(markovstate.acceptanceProbability)
                );
	            
                dataset.write(markovstate.acceptanceProbability);

	            // write out the weights (eg logdetJ, etc. . .)
	            for (auto [key,field] : markovstate.weights) {
	                dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(
                        baseNode+"/weights/"+key,
                        HighFive::DataSpace::From(
                            static_cast <std::complex<NSL::RealTypeOf<Type>>> (field)
                        )
                    );

	                dataset.write(static_cast <std::complex<NSL::RealTypeOf<Type>>> (field));
	            }
	   
	            // write out the markovTime
	            dataset = h5f_.createDataSet<int>(
                    baseNode+"/markovTime",
                    HighFive::DataSpace::From(markovstate.markovTime)
                );
	            dataset.write(markovstate.markovTime);

            // Dealing with non complex types
	        } else {
	            // write out the actionValue
	            HighFive::DataSet dataset = h5f_.createDataSet<Type>(
                    baseNode+"/actVal",
                    HighFive::DataSpace::From(markovstate.actionValue)
                );
	            dataset.write(markovstate.actionValue);

	            // write out the acceptanceProbability
	            dataset = h5f_.createDataSet<Type>(
                    baseNode+"/acceptanceProbability",
                    HighFive::DataSpace::From(markovstate.acceptanceProbability)
                );
	            dataset.write(markovstate.acceptanceProbability);

	            // write out the weights (eg logdetJ, etc. . .)
	            for (auto [key,field] : markovstate.weights) {
	                dataset = h5f_.createDataSet<Type>(
                        baseNode+"/weights/"+key,
                        HighFive::DataSpace::From(field)
                    );
	                dataset.write(field);
	            }

	            // write out the markovTime
	            dataset = h5f_.createDataSet<int>(
                    baseNode+"/markovTime",
                    HighFive::DataSpace::From(markovstate.markovTime)
                );
	            dataset.write(markovstate.markovTime);
	        }
	
            return 0;
        } // write(markovState,node) 

        template <NSL::Concept::isNumber Type> 
        inline int read(NSL::MCMC::MarkovState<Type> &markovstate, const std::string node){
	        // no specific Markov time is given, so find the latest one
	        auto configs = h5f_.getGroup(node).listObjectNames();  // this list all the stored configuration numbers
            
            // this list is not given in ascending order!  Really annoying!  I have to loop over them to find the most recent config. . .
            markovstate.markovTime = std::stoi(configs[0]);
            for (int i=1;i<configs.size();i++){
                if (std::stoi(configs[i])>markovstate.markovTime) {
	                markovstate.markovTime = std::stoi(configs[i]);
                }
            }
	        NSL::Logger::info("Searching for most recent trajectory . . . found {}/{}", node,markovstate.markovTime);
	        this -> read(markovstate, node, markovstate.markovTime);

            return 0;
        } // read(markovstate,node)

        template <NSL::Concept::isNumber Type> 
        inline int read(NSL::MCMC::MarkovState<Type> &markovstate, const std::string node, const int markovTime){
            std::string baseNode = node;

            markovstate.markovTime = markovTime;
	    
	        NSL::Logger::info("Loading in {}",baseNode);

            if constexpr (NSL::is_complex<Type>()) {
                std::complex<NSL::RealTypeOf<Type>> temp; // I need to define a temp variable, since I cannot static_cast within a dataset.read() call
	       
        	    this->read(markovstate.configuration, baseNode); // read in the configuration

	            // read in the actionValue
	            HighFive::DataSet dataset = h5f_.getDataSet(baseNode+"/actVal");
	            dataset.read(temp);
	            markovstate.actionValue = temp;

	            // read in the acceptanceProbability
	            dataset = h5f_.getDataSet(baseNode+"/acceptanceProbability");
	            dataset.read(markovstate.acceptanceProbability);

	            // read in the weights (eg logdetJ, etc. . .)
	            for (auto & [key,field] : markovstate.weights) {
	                dataset = h5f_.getDataSet(baseNode+"/weights/"+key);
	                dataset.read(temp);
	                field = temp;
	            }
	        } else {
        	    this -> read(markovstate.configuration, baseNode); // read in the configuration

	            // read in the actionValue
	            HighFive::DataSet dataset = h5f_.getDataSet(baseNode+"/actVal");
	            dataset.read(markovstate.actionValue);

	            // read in the acceptanceProbability
	            dataset = h5f_.getDataSet(baseNode+"/acceptanceProbability");
	            dataset.read(markovstate.acceptanceProbability);

	            // read in the weights (eg logdetJ, etc. . .)
	            for (auto & [key,field] : markovstate.weights) {
	                dataset = h5f_.getDataSet(baseNode+"/weights/"+key);
	                dataset.read(field);
	            }	
            }
	    
            return 0;
        } // read(markovState, node, markovTime)

        template <NSL::Concept::isNumber Type> 
        inline int write(NSL::Tensor<Type> &tensor_in, const std::string node){
	        std::string DIM("shape");
        	std::vector<NSL::size_t> shape = tensor_in.shape();
	        std::string FORMAT("type");
	        std::string typeID = typeid(Type).name();

            NSL::Device dev = tensor_in.device();

            NSL::Tensor<Type> tensor = tensor_in.to(NSL::CPU()); 

	        if constexpr (NSL::is_complex<Type>()) {
                std::vector<std::complex<NSL::RealTypeOf<Type>>> phi(
                    tensor.data(), 
                    tensor.data()+tensor.numel()
                );

              	HighFive::DataSet dataset = h5f_.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(
                    node,  
                    HighFive::DataSpace::From(phi)
                );
              	dataset.write(phi);
              
	           // now write out the dimension of the tensor as an attribute
              	HighFive::Attribute dim = dataset.createAttribute<int>(DIM,HighFive::DataSpace::From(shape));
              	dim.write(shape);

	           // write out the type
	           HighFive::Attribute form = dataset.createAttribute<std::string>(FORMAT,HighFive::DataSpace::From(typeID));
	           form.write(typeID);
            } else {
                std::vector<Type> phi(tensor.data(), tensor.data()+tensor.numel());

              	HighFive::DataSet dataset = h5f_.createDataSet<Type>(node,  HighFive::DataSpace::From(phi));
              	dataset.write(phi);

              	// now write out the dimension of the tensor as an attribute
              	HighFive::Attribute dim = dataset.createAttribute<int>(DIM,HighFive::DataSpace::From(shape));
              	dim.write(shape);

	           	// write out the type
	            HighFive::Attribute form = dataset.createAttribute<std::string>(FORMAT,HighFive::DataSpace::From(typeID));
	            form.write(typeID);
            }

            // copy back to original device in case the tensor is re used.
            tensor.to(dev);
	    h5f_.flush();  // force writing to disk!            
	        return 0; 
        } // write(tensor,node)

        template <NSL::Concept::isNumber Type> 
        inline int write(const NSL::Configuration<Type> &config, const std::string node){
	        for (auto [key,field] : config) {
	            if (node.back() == '/'){
	                this -> write(field, node+key);
	            } else {
	                this -> write(field, node+"/"+key);
	            }
	        }
            
	        return 0; 
        } // write(config, node)

        template <NSL::Concept::isNumber Type> 
        inline int read(NSL::Tensor<Type> &tensor,const std::string node){
            auto typeID = typeid(Type).name();
            std::string h5type;

            if(h5f_.exist(node)){ // check if the node exists
                HighFive::DataSet dataset = h5f_.getDataSet(node);
	        
	            // first make sure the types match
	            HighFive::Attribute format = dataset.getAttribute("type");
	            format.read(h5type);
	            if(h5type != typeID) {
	              NSL:Logger::warn("Tensor types don't match!  Desire {}, but loading in {}", typeID,h5type);
	            }
	        
              	// first read the attributes to get the shape of the tensor
	            std::vector<NSL::size_t> shape;
	            HighFive::Attribute dim = dataset.getAttribute("shape");
	            dim.read(shape);
	            int numElems = 1;

	            for (NSL::size_t i=0;i<shape.size();i++){ 
                    numElems *= shape[i]; 
                }
	            
	            if(!tensor.defined()){
	                tensor = NSL::Tensor<Type> (numElems);
	            } else {
	                tensor.flatten();  // flatten the array
	                if(tensor.numel() != numElems){    
	                    tensor.resize(numElems);  // Need to resize the array here!
	                }
	            }

                // get device information from the tensor to move the read data to the right
                // places
                NSL::Device dev = tensor.device();

                // get the tensor to the CPU if required
                tensor.to(NSL::CPU());

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

                // copy back to GPU if required;
                tensor.to(dev);

                return 0;
            } else { 
                // node does not exist
                NSL::Logger::error("Error! Node {} doesn't exist!", node); 
                
                return 1;
            }

            // I assume that once phi is 'off the stack', its destructor will be called and its memory released
        } // read(NSL::Tensor, std::string)

        template <NSL::Concept::isNumber Type> 
        inline int read(NSL::Configuration<Type> &config, const std::string node){
	        for (auto & [key,field] : config) {
	            if (node.back() == '/' ){
	                this -> read(field, node+key);
	            } else {
	                this -> read(field, node+"/"+key);
	            }
	        }
        
	        return 0; 
        } // read(config,node)
  
    
    private:
        std::string h5file_;
        File h5f_;
    
}; // class H5IO

} // namespace NSL

#endif // NSL_IO_H5_IO_TPP
