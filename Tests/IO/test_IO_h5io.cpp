#include "../test.hpp"
//#include <highfive/H5DataSet.hpp>
//#include <highfive/H5DataSpace.hpp>
#include "highfive/H5File.hpp"
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
  auto t = NSL::Matrix::Identity<Type>(Nx);    
  auto det = NSL::LinAlg::det(t);

  std::string FILE_NAME("./test_IO.h5");
  std::string DATASET_NAME("configurations/"+std::to_string(0)+"/phi");
    
  // We create an empty HDF5 file, by truncating an existing
  // file if required:
  File h5file(FILE_NAME, File::Truncate);
  
  std::vector<std::complex<double>> phi(Nx*Nt);

  //  std::vector<size_t> dims{Nt*Nx};  
  //  DataSet dataset = h5file.createDataSet<double>(DATASET_NAME, DataSpace(dims));
  //  double phi[Nt*Nx];  
  //  dataset.write(phi);
  
  h5file.createDataSet(DATASET_NAME, phi);
  
  
  REQUIRE( det == static_cast<Type>(1) );

}




