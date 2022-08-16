#include "NSL.hpp"
#include <cmath>

#include "../benchmark.hpp"
#include "FermionMatrix/Impl/hubbardDiag.hpp"
#include "FermionMatrix/Impl/hubbardDiag.tpp"

void bench_exp_cube_M(int nt, int nx){

    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw, est_err=0.0;
    
    //time at each point
    std::vector<double> time_point(Nsweep);

    Timer<std::chrono::nanoseconds> timer("M_time");
    
    for(int i=0; i<Nsweep; i++){
        timer.start();
        M.M(psi);
        timer.stop();
        time_point[i]= timer.get_time();
    }
            
    //find mean
    timings = mean(time_point);
    est_err = err(time_point);

    
    timings = (timings) * pow(10,-9);
    est_err = est_err * pow(10,-9);
    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    est_bw = ((nt*nx*nx*nx * sizeof(NSL::complex<double>) * 9.3 * pow(10,-10))/timings);
    
    std::cout << "Estimated Time [s] for function call to M: "<<timings<<" +/- "<<est_err<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;
    std::string filename;
    filename = "results/results_bench_exp_cube_cpu_M_nx_" + std::to_string(nx) + ".txt";  
    std::ofstream fout; 
    fout.open(filename , std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<" "<<est_err<<std::endl;
    fout.close();
}

void bench_exp_cube_MdaggerM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw, est_err=0.0;
    
    //time at each point
    std::vector<double> time_point(Nsweep);

    Timer<std::chrono::nanoseconds> timer("M_time");
    
    for(int i=0; i<Nsweep; i++){
        timer.start();
        M.MdaggerM(psi);
        timer.stop();
        time_point[i]= timer.get_time();
    }
            
    //find mean
    timings = mean(time_point);
    est_err = err(time_point);

    
    timings = (timings) * pow(10,-9);
    est_err = est_err * pow(10,-9);

    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    est_bw = ((nt*nx*nx*nx * sizeof(NSL::complex<double>) * 9.3 * pow(10,-10))/timings);
    
    std::cout << "Estimated Time [s] for function call to MdaggerM: "<<timings<<" +/- "<<est_err<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;
    std::string filename;
    filename = "results/results_bench_exp_cube_cpu_MdaggerM_nx_" + std::to_string(nx) + ".txt";  
    std::ofstream fout; 
    fout.open(filename , std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<" "<<est_err<<std::endl;
    fout.close();

}

void bench_exp_cube_logDetM(int nt, int nx){
    NSL::Tensor<NSL::complex<double>> psi(nt, nx*nx*nx), phi(nt, nx*nx*nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Cube3D<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardExp M(lat,phi);

    int Nsweep=1000;
    double timings=0.0, est_bw, est_err=0.0;
    
    //time at each point
    std::vector<double> time_point(Nsweep);

    Timer<std::chrono::nanoseconds> timer("M_time");
    
    for(int i=0; i<Nsweep; i++){
        timer.start();
        M.logDetM();
        timer.stop();
        time_point[i]= timer.get_time();
    }
            
    //find mean
    timings = mean(time_point);
    est_err = err(time_point);

    
    timings = (timings) * pow(10,-9);
    est_err = est_err * pow(10,-9);
    std::cout<<"Nt = "<<nt<<" Nx = "<<nx<<std::endl;
    std::cout.precision(17);
    est_bw = ((nt*nx*nx*nx * sizeof(NSL::complex<double>) * 9.3 * pow(10,-10))/timings);
    
    std::cout << "Estimated Time [s] for function call to logDetM: "<<timings<<" +/- "<<est_err<<std::endl;
    std::cout<< "Estimated bandwidth: "<<est_bw<<std::endl;
    std::string filename;
    filename = "results/results_bench_exp_cube_cpu_logDetM_nx_" + std::to_string(nx) + ".txt";  
    std::ofstream fout; 
    fout.open(filename , std::ios::out | std::ios::app);
    fout<<nt<<" "<<nx<<" "<<timings<<" "<<est_bw<<" "<<est_err<<std::endl;
    fout.close();

}
int main(){
    std::vector<int> Nt = {2,4,8,16,32,64,128,256};
    
    for(int i=0; i< Nt.size(); i++){
        
            bench_exp_cube_M(Nt[i],8);
        }
    
    for(int i=0; i< Nt.size(); i++){
        
            bench_exp_cube_MdaggerM(Nt[i],8);
        
    }
    for(int i=0; i< Nt.size(); i++){
        
            bench_exp_cube_logDetM(Nt[i],8);
        
    }
    

}