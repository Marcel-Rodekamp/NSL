#include "NSL.hpp"

int main(){

  NSL::H5IO io("example_io.h5", File::Truncate); // constructor

  NSL::Tensor<NSL::complex<double>> pout(2,3,4); // define tensor to write out
  pout.rand(); // assign random values
  io.write(pout,"tensor"); // write out the tensor
  
  NSL::Tensor<NSL::complex<double>> pin(3,4); // define tensor to store 
  io.read(pin,"tensor"); // read in the tensor

  // now do something similar for a configuration
  NSL::Tensor<NSL::complex<double>> phi_out(3,3);
  NSL::Configuration<NSL::complex<double>> config_out{{"phi", phi_out}};
  config_out["phi"].rand(); // assign random values
  io.write(config_out, "config/0"); // write out configuration

  // now read in a configuration
  NSL::Tensor<NSL::complex<double>> phi_in(2,3);
  NSL::Configuration<NSL::complex<double>> config_in{{"phi", phi_in}};
  io.read(config_in,"config/0");

  // now do something similar for a markov state
  NSL::MCMC::MarkovState<NSL::complex<double>> markovstate;
  NSL::Tensor<NSL::complex<double>> phi(3,3);
  NSL::Configuration<NSL::complex<double>> config{{"phi", phi}};
  config["phi"].rand(); // assign random values
  markovstate.configuration = config;
  markovstate.actionValue = 1000.0;
  markovstate.acceptanceProbability = .6667;
  markovstate.markovTime = 0;
  io.write(markovstate,"markovstate");

  
  return 0;
}

