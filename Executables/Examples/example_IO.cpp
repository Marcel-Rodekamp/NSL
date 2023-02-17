#include "NSL.hpp"

int main(){

  NSL::H5IO io("example_io.h5");
  NSL::Tensor<NSL::complex<double>> pout(2,3,4);

  pout.rand();

  std::cout << pout.shape() << std::endl;

  io.write(pout,"config/0/phi");
  
  NSL::Tensor<NSL::complex<double>> pin(3,4);
  io.read(pin,"config/0/phi");

  std::cout << pin.shape() << std::endl;
  
  return 0;
}

