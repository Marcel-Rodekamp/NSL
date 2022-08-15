#include "NSL.hpp"
#include <cmath>

#include "../benchmark.hpp"
#include "FermionMatrix/Impl/hubbardDiag.hpp"
#include "FermionMatrix/Impl/hubbardDiag.tpp"
#include <fstream>

void bench_exp_ring_M(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(NSL::GPU(),nt,nx), phi(NSL::GPU(),nt,nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Ring<NSL::complex<double>> lat(nx);
    lat.to(NSL::GPU());
    
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw;

    Timer<std::chrono::nanoseconds> timer("M_time");
    timer.start();
    for(int i=0; i<Nsweep; i++){
        M.M(psi);
    }
    timer.stop();
    timings= timer.get_time();
    timings = ((timings)/Nsweep) * pow(10,-9);

    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    est_bw = ((nt*nx * sizeof(NSL::complex<double>) * pow(10,-9))/timings);
    std::cout << "Estimated Time [s] for function call to M: "<<timings<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;
    
    std::string filename;
    filename = "results_gpu/results_bench_exp_ring_gpu_M_nx_" + std::to_string(nx) + ".txt";  
    std::ofstream fout; 
    fout.open(filename , std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<std::endl;
    fout.close();

}

void bench_exp_ring_MdaggerM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(NSL::GPU(),nt,nx), phi(NSL::GPU(),nt,nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Ring<NSL::complex<double>> lat(nx);
    lat.to(NSL::GPU());
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw;

    Timer<std::chrono::nanoseconds> timer("M_time");
    timer.start();
    for(int i=0; i<Nsweep; i++){
        M.MdaggerM(psi);
    }
    timer.stop();
    timings= timer.get_time();
    timings = ((timings)/Nsweep) * pow(10,-9);
    
    est_bw = ((nt*nx * sizeof(NSL::complex<double>) * pow(10,-9))/timings);
    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    std::cout << "Estimated Time [s] for function call to MdaggerM: "<<timings<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;

    std::string filename;
    filename = "results_gpu/results_bench_exp_ring_gpu_MdaggerM_nx_" + std::to_string(nx) + ".txt"; 
    std::ofstream fout; 
    fout.open(filename, std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<std::endl;
    fout.close();

}

void bench_exp_ring_logDetM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(NSL::GPU(),nt,nx), phi(NSL::GPU(),nt,nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Ring<NSL::complex<double>> lat(nx);
    lat.to(NSL::GPU());
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw;

    Timer<std::chrono::nanoseconds> timer("M_time");
    timer.start();
    for(int i=0; i<Nsweep; i++){
        M.logDetM();
    }
    timer.stop();
    timings= timer.get_time();
    timings = ((timings)/Nsweep) * pow(10,-9);
    est_bw = ((nt*nx * sizeof(NSL::complex<double>) * pow(10,-9))/timings);
    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    std::cout << "Estimated Time [s] for function call to logDetM: "<<timings<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;

    std::string filename;
    filename = "results/results_bench_exp_ring_logDetM_nx_" + std::to_string(nx) + "_gpu.txt" ;
    std::ofstream fout; 
    fout.open(filename, std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<std::endl;
    fout.close();

}
int main(){

    std::vector<int> Nt = {2,4,8,16,32,64,128,256};
    //std::vector<int> Nt = {128,256,512,1024};
    //for(int i=0; i< Nt.size(); i++){       
    //        bench_exp_ring_M(Nt[i],128);
    //}

    //
    //for(int i=0; i< Nt.size(); i++){       
    //        bench_exp_ring_MdaggerM(Nt[i],128);
    //}

    
    for(int i=0; i< Nt.size(); i++){       
            bench_exp_ring_logDetM(Nt[i],8);
    }

    /*
    std::vector<int> Nt = {16,32,64,256}, Nx = {2,4,8,16,32};
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_exp_ring_M(Nt[i],Nx[j]);
        }
    }
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_exp_ring_MdaggerM(Nt[i],Nx[j]);
        }
    }
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_exp_ring_logDetM(Nt[i],Nx[j]);
        }
    }
    */

}
