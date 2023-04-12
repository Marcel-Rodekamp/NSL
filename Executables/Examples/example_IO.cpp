#include "NSL.hpp"

int main(){

  NSL::H5IO io("example_io.h5", NSL::File::Truncate); // constructor

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

  // now do something similar for a markov state defined with floats
  NSL::MCMC::MarkovState<float> markovstateF;
  NSL::Tensor<float> phiF(18,32);
  NSL::Configuration<float> configF{{"phi", phiF}};
  configF["phi"].rand(); // assign random values
  markovstateF.configuration = configF;
  markovstateF.actionValue =  1000.0;
  markovstateF.acceptanceProbability = .6667;
  markovstateF.markovTime = 0;
  io.write(markovstateF,"markovstate");
  
  // now do something similar for a markov state defined with doubles
  NSL::MCMC::MarkovState<double> markovstateD;
  NSL::Tensor<double> phiD(18,32);
  NSL::Configuration<double> configD{{"phi", phiD}};
  configD["phi"].rand(); // assign random values
  markovstateD.configuration = configD;
  markovstateD.actionValue =  1000.0;
  markovstateD.acceptanceProbability = .6667;
  markovstateD.markovTime = 1;
  io.write(markovstateD,"markovstate");

  // now do something similar for a markov state defined with complex floats
  NSL::MCMC::MarkovState<NSL::complex<float>> markovstateCF;
  NSL::Tensor<NSL::complex<float>> phiCF(18,32);
  NSL::Configuration<NSL::complex<float>> configCF{{"phi", phiCF}};
  configCF["phi"].rand(); // assign random values
  markovstateCF.configuration = configCF;
  markovstateCF.actionValue = NSL::complex<float> (1000.0,0.0);
  markovstateCF.acceptanceProbability = .6667;
  markovstateCF.markovTime = 2;
  io.write(markovstateCF,"markovstate");

  // now do something similar for a markov state defined with complex doubles
  NSL::MCMC::MarkovState<NSL::complex<double>> markovstateCD;
  NSL::Tensor<NSL::complex<double>> phiCD(18,32);
  NSL::Configuration<NSL::complex<double>> configCD{{"phi", phiCD}};
  configCD["phi"].rand(); // assign random values
  markovstateCD.configuration = configCD;
  markovstateCD.actionValue = NSL::complex<double> (1000.0,0.0);
  markovstateCD.acceptanceProbability = .6667;
  markovstateCD.markovTime = 3;
  io.write(markovstateCD,"markovstate");

  // now read in the most recent markovstate
  io.read(markovstateCD,"markovstate");

  // now read in a specific markovstate
  io.read(markovstateD,"markovstate",1);
  
  return 0;
}

