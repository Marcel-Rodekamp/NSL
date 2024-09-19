#include "NSL.hpp"
#include <chrono>

// test correctness and speed of each function in the fermion matrix
void compare_old_new(int L, int NT){
    typedef NSL::complex<double> Type;
    NSL::complex<double> I{0,1};

    NSL::Lattice::Square<Type> lattice({L,L});
    int NX = L*L;

    std::cout << NX << " " << NT << std::endl;
    // Define the parameters
    NSL::Parameter params;
    params["Nt"] = NT;
    params["beta"] = 1.0*NT;
    params["mu"] = 0.0;
    params["U"] = 3.0;

    NSL::Tensor<Type> phi(NT,NX), psi(NT,NX);
    phi.randn();
    psi.randn();

    // Define the fermion matrix
    NSL::FermionMatrix::HubbardExp<Type,NSL::Lattice::Square<Type>> FM(lattice,params);
    FM.populate(phi);

    NSL::Tensor<Type> result, result_new;
    std::chrono::steady_clock::time_point begin, end;
    
    // test correctness of M
    result = FM.M(psi);
    result_new = FM.M_new(psi);
    std::cout << "M vs M_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    // measure speed of M
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.M(psi);
    }
    end = std::chrono::steady_clock::now();
    auto M_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "M: \t" << M_time << std::endl;

    // measure speed of M_new
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.M_new(psi);
    }
    end = std::chrono::steady_clock::now();
    auto M_new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "M_new: \t" << M_new_time << std::endl;
    std::cout << "M_speedup: " << M_time*100/M_new_time << "%" << std::endl;

    // test correctness of Mdagger
    result = FM.Mdagger(psi);
    result_new = FM.Mdagger_new(psi);
    std::cout << "Mdagger vs Mdagger_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    // measure speed of Mdagger
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.Mdagger(psi);
    }
    end = std::chrono::steady_clock::now();
    auto Mdagger_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "Mdagger: \t" << Mdagger_time << std::endl;

    // measure speed of Mdagger_new
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.Mdagger_new(psi);
    }
    end = std::chrono::steady_clock::now();
    auto Mdagger_new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "Mdagger_new: \t" << Mdagger_new_time << std::endl;
    std::cout << "Mdagger_speedup: \t" << Mdagger_time*100/Mdagger_new_time << "%" << std::endl;

    // test correctness of MMdagger
    result = FM.MMdagger(psi);
    result_new = FM.MMdagger_new(psi);
    std::cout << "MMdagger vs MMdagger_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    // measure speed of MMdagger
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.MMdagger(psi);
    }
    end = std::chrono::steady_clock::now();
    auto MMdagger_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "MMdagger: \t" << MMdagger_time << std::endl;

    // measure speed of MMdagger_new
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.MMdagger_new(psi);
    }
    end = std::chrono::steady_clock::now();
    auto MMdagger_new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "MMdagger_new: \t" << MMdagger_new_time << std::endl;
    std::cout << "MMdagger_speedup: \t" << MMdagger_time*100/MMdagger_new_time << "%" << std::endl;

    // test correctness of MdaggerM
    result = FM.MdaggerM(psi);
    result_new = FM.MdaggerM_new(psi);
    std::cout << "MdaggerM vs MdaggerM_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    // measure speed of MdaggerM
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.MdaggerM(psi);
    }
    end = std::chrono::steady_clock::now();
    auto MdaggerM_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "MdaggerM: \t" << MdaggerM_time << std::endl;

    // measure speed of MdaggerM_new
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.MdaggerM_new(psi);
    }
    end = std::chrono::steady_clock::now();
    auto MdaggerM_new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "MdaggerM_new: \t" << MdaggerM_new_time << std::endl;
    std::cout << "MdaggerM_speedup: \t" << MdaggerM_time*100/MdaggerM_new_time << "%" << std::endl;

    // test correctness of dMdPhi
    result = FM.dMdPhi(psi,psi);
    result_new = FM.dMdPhi_new(psi,psi);
    std::cout << "dMdPhi vs dMdPhi_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    // measure speed of dMdPhi
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.dMdPhi(psi,psi);
    }
    end = std::chrono::steady_clock::now();
    auto dMdPhi_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "dMdPhi: \t" << dMdPhi_time << std::endl;

    // measure speed of dMdPhi_new
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5000; i++){
        FM.dMdPhi_new(psi,psi);
    }
    end = std::chrono::steady_clock::now();
    auto dMdPhi_new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "dMdPhi_new: \t" << dMdPhi_new_time << std::endl;
    std::cout << "dMdPhi_speedup: \t" << dMdPhi_time*100/dMdPhi_new_time << "%" << std::endl;

}

int main(int argc, char* argv[]){
    for (int nx = 10; nx < 13; nx+=2){
        for (int nt = 8; nt < 129; nt*=2){
            compare_old_new(nx,nt);
        }
    }
}
