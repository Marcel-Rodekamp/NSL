#include "../test.hpp"

COMPLEX_NSL_TEST_CASE( "fermionMatrixU1Wilson: M", "[fermionMatrixU1,Wilson, M, dense]" ) {
    const NSL::size_t Nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t Nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    std::string devRepr = GENERATE( "CPU", "GPU" );
    NSL::Device dev(devRepr);

    TestType I{0,1};
    NSL::size_t dim = 2;
    NSL::size_t mu_t = 0;
    NSL::size_t mu_x = 1;
    TestType bareMass = 0;

    INFO(fmt::format("Nt={}; Nx={}; dim={}; dev={}", Nt,Nx,dim,devRepr));

    // This tests the free theory against an exact calculation
    NSL::Tensor<TestType> psi(dev,Nt,Nx,dim);
    psi.randn();
    //psi(0,0,0) = 1;

    NSL::Lattice::Square<TestType> lattice({Nt,Nx});
    lattice.to(dev);

    NSL::Parameter params;
    params["Nt"]= Nt ;
    params["Nx"]=Nx ;
    params["dim"]=dim ;
    params["bare mass"]= bareMass ;
    params["device"] =dev ;
    NSL::Tensor<TestType> U(dev,Nt,Nx,dim); U = 1;
 
    NSL::FermionMatrix::U1::Wilson<TestType> wfm(lattice,params);
    wfm.populate(U);

    auto impleResult = wfm.M(psi);

    // copy result to cpu to compate
    impleResult.to(true,NSL::CPU());

    // as U = 1; The components are defined as 
    // analyResult stores the analytical result of
    // \sum_{y=0}^{V-1} \sum_{beta=0}^{D-1} M_{xy;\alpha,\beta} \cdot \psi^{\beta}(y) 
    NSL::Tensor<TestType> analyResult(dev,Nt,Nx,dim);

    // alpha = 0
    analyResult(NSL::Ellipsis(),0) = -0.5*(
        // mu = mu_t
           NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        -  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        // mu = mu_x
        +  NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),0)
        +I*NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),0)
        -I*NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),1)
    );

    // alpha = 1
    analyResult(NSL::Ellipsis(),1) = -0.5*(
        // mu = mu_t
        -  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        // mu = mu_x
        -I*NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),1)
        +I*NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),1)
    );

    analyResult += (bareMass+dim)*psi;

    // copy result to cpu to compate
    analyResult.to(true,NSL::CPU());

    //std::cout << analyResult.real() << std::endl;
    //std::cout << impleResult.real() << std::endl;
    //std::cout << "==============================" << std::endl;
    //std::cout << analyResult.imag() << std::endl;
    //std::cout << impleResult.imag() << std::endl;

    // now these 2 should agree
    for(NSL::size_t t = 0; t < Nt; ++t){
    for(NSL::size_t x = 0; x < Nx; ++x){
    for(NSL::size_t a = 0; a < dim; ++a){
        INFO(fmt::format("t={}; x={}; a={}",t,x,a));
        INFO(fmt::format("analytic={}+i{}",NSL::real(analyResult(t,x,a)),NSL::imag(analyResult(t,x,a))));
        INFO(fmt::format("algorith={}+i{}",NSL::real(impleResult(t,x,a)),NSL::imag(impleResult(t,x,a))));
        
        REQUIRE(almost_equal( analyResult(t,x,a),impleResult(t,x,a) ));
    }}}

    // We can now test gauge invariance 
    U = NSL::LinAlg::exp(I*NSL::randn<NSL::RealTypeOf<TestType>>(dev,Nt,Nx,dim));
    NSL::Tensor<TestType> phase(dev,Nt,Nx,dim);
    //phase = NSL::LinAlg::exp(I*NSL::randn<NSL::RealTypeOf<TestType>>(dev,Nt,Nx,dim));
    phase(NSL::Ellipsis()) = NSL::LinAlg::exp( I * TestType(0.1) );

    wfm.populate(U);

    auto ppsi = phase*psi;

    TestType Mpsi = (psi.conj()*wfm.M( psi )).sum();
    TestType MpPsi= (ppsi.conj()*wfm.M( ppsi )).sum();

    REQUIRE(almost_equal(Mpsi, MpPsi));
}

COMPLEX_NSL_TEST_CASE( "fermionMatrixU1Wilson: Mdagger", "[fermionMatrixU1,Wilson, Mdagger, dense]" ) {
    const NSL::size_t Nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t Nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    std::string devRepr = GENERATE( "CPU", "GPU" );
    NSL::Device dev(devRepr);

    TestType I{0,1};
    NSL::size_t dim = 2;
    NSL::size_t mu_t = 0;
    NSL::size_t mu_x = 1;
    TestType bareMass = 0;

    // This tests the free theory against an exact calculation
    NSL::Tensor<TestType> psi(dev,Nt,Nx,dim);
    psi.randn();
    //psi(0,0,0) = 1;

    NSL::Lattice::Square<TestType> lattice({Nt,Nx});
    lattice.to(dev);

    NSL::Parameter params;
    params[ "Nt"]= Nt;
    params["Nx"]=Nx;
    params[ "dim"]= dim ;
    params[ "bare mass"]= bareMass ;
    params[ "device"]= dev ;
    NSL::Tensor<TestType> U(dev,Nt,Nx,dim); U = 1;
 
    NSL::FermionMatrix::U1::Wilson<TestType> wfm(lattice,params);
    wfm.populate(U);

    auto impleResult = wfm.Mdagger(psi);

    // copy result to cpu to compate
    impleResult.to(true,NSL::CPU());

    // as U = 1; The components are defined as 
    // analyResult stores the analytical result of
    // \sum_{y=0}^{V-1} \sum_{beta=0}^{D-1} M_{xy;\alpha,\beta} \cdot \psi^{\beta}(y) 
    NSL::Tensor<TestType> analyResult(dev,Nt,Nx,dim);

    // alpha = 0
    analyResult(NSL::Ellipsis(),0) = -0.5*(
        // mu = mu_t
           NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        -  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        // mu = mu_x
        +  NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),0)
        +I*NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),0)
        -I*NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),1)
    );

    // alpha = 1
    analyResult(NSL::Ellipsis(),1) = -0.5*(
        // mu = mu_t
        -  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,+1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        +  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,-1,mu_t,TestType(-1))(NSL::Ellipsis(),1)
        // mu = mu_x
        -I*NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,+1,mu_x)(NSL::Ellipsis(),1)
        +I*NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),0)
        +  NSL::LinAlg::shift(psi,-1,mu_x)(NSL::Ellipsis(),1)
    );

    analyResult += (bareMass+dim)*psi;

    // copy result to cpu to compate
    analyResult.to(true,NSL::CPU());

    //std::cout << analyResult.real() << std::endl;
    //std::cout << impleResult.real() << std::endl;
    //std::cout << "==============================" << std::endl;
    //std::cout << analyResult.imag() << std::endl;
    //std::cout << impleResult.imag() << std::endl;

    // now these 2 should agree
    for(NSL::size_t t = 0; t < Nt; ++t){
    for(NSL::size_t x = 0; x < Nx; ++x){
    for(NSL::size_t a = 0; a < dim; ++a){
        INFO(fmt::format("t={}; x={}; a={}",t,x,a));
        INFO(fmt::format("analytic={}+i{}",NSL::real(analyResult(t,x,a)),NSL::imag(analyResult(t,x,a))));
        INFO(fmt::format("algorith={}+i{}",NSL::real(impleResult(t,x,a)),NSL::imag(impleResult(t,x,a))));
        
        REQUIRE(almost_equal( analyResult(t,x,a),impleResult(t,x,a) ));
    }}}

    // We can now test gauge invariance 
    U = NSL::LinAlg::exp(I*NSL::randn<NSL::RealTypeOf<TestType>>(dev,Nt,Nx,dim));
    NSL::Tensor<TestType> phase(dev,Nt,Nx,dim);
    //phase = NSL::LinAlg::exp(I*NSL::randn<NSL::RealTypeOf<TestType>>(dev,Nt,Nx,dim));
    phase(NSL::Ellipsis()) = NSL::LinAlg::exp( I * TestType(0.1) );

    wfm.populate(U);

    auto ppsi = phase*psi;

    TestType Mpsi = (psi.conj()*wfm.Mdagger( psi )).sum();
    TestType MpPsi= (ppsi.conj()*wfm.Mdagger( ppsi )).sum();

    REQUIRE(almost_equal(Mpsi, MpPsi));

}
