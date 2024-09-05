#include "NSL.hpp"
#include <chrono>

// test correctness and speed of each function in the fermion matrix

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;
    NSL::complex<double> I{0,1};


    NSL::Lattice::Square<Type> lattice({20,20});
    int NX = lattice.sites();
    int NT = 32;
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
    // measure operations with profiler
    result = FM.M(psi);
    result_new = FM.M_new(psi);
    std::cout << "M vs M_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.M(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "M: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.M_new(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "M_new: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    // test correctness of Mdagger
    result = FM.Mdagger(psi);
    result_new = FM.Mdagger_new(psi);
    std::cout << "Mdagger vs Mdagger_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.Mdagger(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Mdagger: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.Mdagger_new(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Mdagger_new: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    // test correctness of MMdagger
    result = FM.MMdagger(psi);
    result_new = FM.MMdagger_new(psi);
    std::cout << "MMdagger vs MMdagger_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.MMdagger(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "MMdagger: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.MMdagger_new(psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "MMdagger_new: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    // test correctness of dMdPhi
    result = FM.dMdPhi(psi,psi);
    result_new = FM.dMdPhi_new(psi,psi);
    std::cout << "dMdPhi vs dMdPhi_new err: " << ((result - result_new)*NSL::LinAlg::conj(result - result_new)).sum() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.dMdPhi(psi,psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "dMdPhi: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++){
        FM.dMdPhi_new(psi,psi);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "dMdPhi_new: \t" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}
