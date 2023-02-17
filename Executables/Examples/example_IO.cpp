#include "NSL.hpp"

template<typename Type>
void test_h5io(const NSL::size_t & Nt, const NSL::size_t & Nx){

  std::string FILE_NAME("./test_IO.h5");
  std::string DATASET_NAME("configurations/"+std::to_string(0)+"/phi/"+typeid(Type).name());

  NSL::H5IO h5(FILE_NAME);
  
  // create a random tensor array
  auto pout = NSL::Tensor<Type>(Nt,Nx).rand();

  // write it into the h5 file
  h5.write(pout,DATASET_NAME);

  NSL::Tensor<Type> pin;
  // read in the same data
  h5.read(pin, DATASET_NAME);

  //  REQUIRE( (pin == pout).all() );
  
}

int main(){

  NSL::H5IO io("example_io.h5");
  NSL::Tensor<NSL::complex<double>> tensor(32*18);

  tensor.rand();

  std::cout << tensor << std::endl;

  //  io.write(tensor,"config/0/phi");
  test_h5io<NSL::complex<double>>(32,18);

  
  return 0;
}

