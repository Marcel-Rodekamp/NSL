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

  std::string FILE_NAME("./test_IO.h5");
  std::string DATASET_NAME("configurations/"+std::to_string(0)+"/phi");

  NSL::H5IO h5(FILE_NAME);
  
  // create a random tensor array
  auto pp = NSL::Tensor<Type>(Nt, Nx).rand();

  // write it into the h5 file
  h5.write(pp,DATASET_NAME);

  
  
}



