#include "NSL.hpp"
#include <cmath>

#include "../benchmark.hpp"
#include "FermionMatrix/Impl/hubbardDiag.hpp"
#include "FermionMatrix/Impl/hubbardDiag.tpp"

void bench_diag_cube_M(int nt, int nx){

    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardDiag M(lat,phi);

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
    std::cout << "Estimated Time [s] for function call to M: "<<timings<<std::endl;
    
    std::ofstream fout; 
    fout.open("results_bench_diag_cube_M.txt", std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<" "<<timings<<std::endl;
    fout.close();

}

void bench_diag_cube_MdaggerM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardDiag M(lat,phi);

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

    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    std::cout << "Estimated Time [s] for function call to MdaggerM: "<<timings<<std::endl;
    
    std::ofstream fout; 
    fout.open("results_bench_diag_cube_MdaggerM.txt", std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<" "<<timings<<std::endl;
    fout.close();

}

void bench_diag_cube_logDetM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardDiag M(lat,phi);

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

    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    std::cout << "Estimated Time [s] for function call to logDetM: "<<timings<<std::endl;
    
    std::ofstream fout; 
    fout.open("results_bench_diag_cube_logDetM.txt", std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<" "<<timings<<std::endl;
    fout.close();

}
int main(){
    std::vector<int> Nt = {16,32,64,256}, Nx = {2,4,8,16,32};
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_diag_cube_M(Nt[i],Nx[j]);
        }
    }
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_diag_cube_MdaggerM(Nt[i],Nx[j]);
        }
    }
    for(int i=0; i< Nt.size(); i++){
        for(int j=0; j<Nx.size(); j++){
            bench_diag_cube_logDetM(Nt[i],Nx[j]);
        }
    }
    

}