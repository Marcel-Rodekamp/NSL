#include "../test.hpp"
#include <highfive/H5File.hpp>
#include <iostream>

template<typename Type>
void test_h5io(const NSL::size_t & Nt, const NSL::size_t & Nx);

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "IO: read/write to h5 file", "[IO,read/write,default]" ) {
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
  std::string DATASET_NAME("configurations/"+std::to_string(0)+"/"+typeid(Type).name()+"/phi");

  NSL::H5IO h5(FILE_NAME, NSL::File::Truncate);
  
  // create a random tensor array
  auto pout = NSL::Tensor<Type>(Nt, Nx).rand();

  // write it into the h5 file
  h5.write(pout,DATASET_NAME);

  NSL::Tensor<Type> pin;
  // read in the same data with a "null" tensor
  h5.read(pin, DATASET_NAME);

  REQUIRE( (pin == pout).all() );

  NSL::Tensor<Type> pin_d(2,3,4);
  // read in the same data, but now using a previously allocated tensor
  h5.read(pin_d, DATASET_NAME);

  REQUIRE( (pin_d == pout).all() );

  // store in a configuration
  NSL::Configuration<NSL::complex<double>> config_out{{"phi", pout}};
  config_out["phi"].rand(); // re-assign random values
  h5.write(config_out, "configurations/"+std::to_string(1)+"/"+typeid(Type).name());

  // read in the same configuration
  NSL::Configuration<NSL::complex<double>> config_in{{"phi", pin}};
  h5.read(config_in, "configurations/"+std::to_string(1)+"/"+typeid(Type).name());

  REQUIRE( (pin == pout).all() );
}




