#include "../test.hpp"
//#include <highfive/H5Attribute.hpp>
#include <highfive/H5File.hpp>
#include <iostream>

using namespace HighFive;

template<typename Type>
void test_h5io(const NSL::size_t & Nt, const NSL::size_t & Nx);

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: determinant of identity", "[LinAlg,det,default]" ) {
  const NSL::size_t Nt = 32; //GENERATE(32);
  const NSL::size_t Nx = 18; //GENERATE(18);
  test_h5io<TestType>(Nt,Nx);
}

//=======================================================================
// Implementation Details
//=======================================================================

template<typename Type>
void test_h5io(const NSL::size_t & Nt, const NSL::size_t & Nx){
  //auto t = NSL::Matrix::Identity<Type>(Nx);    
  //  auto det = NSL::LinAlg::det(t);

  std::string FILE_NAME("./test_IO.h5");
  std::string DATASET_NAME("configurations/"+std::to_string(0)+"/phi");
  std::string DIM("Nt, Nx");

  NSL::H5IO h5(FILE_NAME);
  
  // We create an empty HDF5 file, by truncating an existing
  // file if required:
  //  File h5file(FILE_NAME, File::Truncate);

  //  NSL::Tensor <Type> pp(Nt,Nx).rand();
  auto pp = NSL::Tensor<Type>(Nt, Nx).rand();

  h5.write(pp,DATASET_NAME);

  /*
  
  auto ppflat = pp.flatten();
  
  //std::cout << ppflat[0] << std::endl;

  std::vector<int> NtNx(2);

  NtNx[0] = Nt;
  NtNx[1] = Nx;
  
  if constexpr (NSL::is_complex<Type>()) {
    std::vector<std::complex<NSL::RealTypeOf<Type>>> phi(pp.data(), pp.data()+pp.numel());

    DataSet dataset = h5file.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(DATASET_NAME,  DataSpace::From(phi));
    dataset.write(phi);

    // now write out the dimension of the tensor as an attribute
    Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(NtNx));
    dim.write(NtNx);
  } else {
    std::vector<Type> phi(pp.data(), pp.data()+pp.numel());

    DataSet dataset = h5file.createDataSet<Type>(DATASET_NAME,  DataSpace::From(phi));
    dataset.write(phi);

    // now write out the dimension of the tensor as an attribute
    Attribute dim = dataset.createAttribute<int>(DIM,DataSpace::From(NtNx));
    dim.write(NtNx);
  }
  */

  
  
  //REQUIRE( det == static_cast<Type>(1) );

  /* highfive now allows the output of multi-dimensional objects, but
     it is not clear if we will want to do something like this. The 
     code below is a segment of what would be needed to write out a multi-
     dim object.
  */
  //  std::vector<size_t> dims{Nt*Nx};  
  //  DataSet dataset = h5file.createDataSet<double>(DATASET_NAME, DataSpace(dims));
  //  double phi[Nt*Nx];  
  //  dataset.write(phi);

  
}




