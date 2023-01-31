#include "NSL.hpp"

int main(){

  NSL::H5IO io("example_io.h5");
  NSL::Tensor<NSL::complex<double>> tensor(2,2);

  tensor.rand();

  std::cout << tensor << std::endl;

  io.write(tensor,"config/0/phi");
  
  return 0;
}
