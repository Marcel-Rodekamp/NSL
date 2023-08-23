#include "../test.hpp"
#include <highfive/H5File.hpp>
#include <iostream>

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_Hubbard_force();

template<typename Type>
void test_U1_force();

template<typename Type>
void test_U1_Wilson_PseudoFermion_force();

template<typename Type>
void test_U1_Wilson_dMdPhi();

template<typename Type>
void test_U1_Wilson_dMdaggerdPhi();

// =============================================================================
// Test Cases
// =============================================================================

COMPLEX_NSL_TEST_CASE( "Force", "[Action,force,default]" ) {
    //INFO("Hubbard Exponential");
    //test_Hubbard_force<TestType,NSL::FermionMatrix::HubbardExp>();
    //INFO("Hubbard Diagonal");
    //test_Hubbard_force<TestType,NSL::FermionMatrix::HubbardDiag>();
    //INFO("U1 Gauge Action");
    //test_U1_force<TestType>();
    //INFO("U1 Wilson Fermion: Pseudofermion");
    //test_U1_Wilson_PseudoFermion_force<TestType>();

    INFO("U1 Wilson Fermion: dM/dPhi");
    test_U1_Wilson_dMdPhi<TestType>();
    INFO("U1 Wilson Fermion: dMdagger/dPhi");
    test_U1_Wilson_dMdaggerdPhi<TestType>();
}

//=======================================================================
// Implementation Details
//=======================================================================

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_Hubbard_force(){
    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type U    = 3.0;
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a ring with {} sites.", NSL::real(beta), Nt, NSL::real(U), Nx);

    NSL::Parameter params;
    params.addParameter<Type>("beta",beta);
    params.addParameter<Type>("U",U);
    params.addParameter<Type>("mu",0);
    params.addParameter<NSL::size_t>("Nt",Nt);
    params.addParameter<decltype(lattice)>("lattice", lattice);

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),HFM<Type,decltype(lattice)>> S_fermion(params);

    // Initialize the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].randn();
    config["phi"].imag() = 0; // use purely real fields
    
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");

    NSL::Configuration<Type> gradS{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    // This is how one computes the grad of the action
    gradS = S.grad(config);

    REQUIRE( (config["phi"].imag() == S.force(config)["phi"].imag() ).all() );  // the force should all real (when chemical potential is zero) 
    
    NSL::RealTypeOf<Type> epsilon = 0.0001;
    auto S_val = S(config);
    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            NSL::Configuration<Type> configE(config,true);
	        configE["phi"](t,i) += epsilon;
            
            auto fin_diff = (S(configE)-S_val)/epsilon;
            auto err = NSL::real(NSL::LinAlg::abs(fin_diff-gradS["phi"](t,i)));
            int fin_diff_order = getMatchingDigits(err);
            int eps_order = getMatchingDigits(epsilon);

            std::string repr = fmt::format("t={},x={}",t,i);
            INFO( repr );
            repr = fmt::format("ΔS = {}",NSL::to_string(fin_diff));
            INFO( repr );
            repr = fmt::format("∂S = {}",NSL::to_string(gradS["phi"](t,i)));
            INFO( repr );
            repr = fmt::format("err= {}",err);
            INFO( repr );
            repr = fmt::format(" ε = {}",epsilon);
            REQUIRE( fin_diff_order >= eps_order );

	        //  Note! this test will ALWAYS fail for complex<float>.  The logDetM routine is too imprecise in this case when calculating the finite differencing!
        }
    }
}

template<typename Type>
void test_U1_force(){
    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type m    = 1.0;
    //    Number of time slices
    NSL::size_t Nt = 2;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 1;
    //    Dimension of the System
    NSL::size_t dim = 2;

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1}; 

    NSL::Parameter params;
    params.addParameter<Type>("beta",beta);
    params.addParameter<Type>("bare mass",m);
    params.addParameter<NSL::size_t>("Nt",Nt);
    params.addParameter<NSL::size_t>("Nx",Nx);
    params.addParameter<NSL::size_t>("dim",dim);

    NSL::Action::U1::WilsonGaugeAction<Type> S_gauge(params);

    // Initialize the action
    NSL::Action::Action S = S_gauge;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx,dim)}
    };
    config["phi"] = NSL::LinAlg::exp(
        I * NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim) 
    );

    // This is how one computes the grad of the action
    NSL::Configuration<Type> gradS = S.grad(config);

    REQUIRE( almost_equal(NSL::RealTypeOf<Type>(0), NSL::imag(S(config))) ); 
    REQUIRE( (0 == S.force(config)["phi"].imag() ).all() ); 

    REQUIRE( ((gradS + S.force(config))["phi"] == 0 ).all() );

    NSL::RealTypeOf<Type> epsilon = 0.0001;
    auto S_val = S(config);
    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            NSL::Configuration<Type> configE(config,true);
	        configE["phi"](t,i) *= NSL::LinAlg::exp(I*epsilon);
            
            auto fin_diff = (S(configE)-S_val)/epsilon;
            auto err = NSL::real(NSL::LinAlg::abs(fin_diff-gradS["phi"](t,i)));
            std::string repr = fmt::format("err= {}",err);
            INFO( repr );
            int fin_diff_order = getMatchingDigits(err);
            int eps_order = getMatchingDigits(epsilon);

            repr = fmt::format("t={},x={}",t,i);
            INFO( repr );
            repr = fmt::format("ΔS = {}",NSL::to_string(fin_diff));
            INFO( repr );
            repr = fmt::format("∂S = {}",NSL::to_string(gradS["phi"](t,i)));
            INFO( repr );
            repr = fmt::format(" ε = {}",epsilon);
            REQUIRE( fin_diff_order >= eps_order );

	        //  Note! this test will ALWAYS fail for complex<float>.
        }
    }
}

template<typename Type>
void test_U1_Wilson_PseudoFermion_force(){
    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type m    = 2.0;
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 8;
    //    Dimension of the System
    NSL::size_t dim = 2;
    // for larger lattices this fails as the finite difference fails

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1}; 

    NSL::Parameter params;
    params.addParameter<Type>("beta",beta);
    params.addParameter<Type>("bare mass",m);
    params.addParameter<NSL::size_t>("Nt",Nt);
    params.addParameter<NSL::size_t>("Nx",Nx);
    params.addParameter<NSL::size_t>("dim",dim);
    params.addParameter<NSL::Device>("device",NSL::CPU());

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    params.addParameter<NSL::Lattice::Square<Type>>("lattice",lattice);
    
    // Initialize the action
    NSL::Action::Action S = NSL::Action::PseudoFermionAction<
                                Type,decltype(lattice), NSL::FermionMatrix::U1::Wilson<Type>
                            >(params)
                         // + NSL::Action::U1::WilsonGaugeAction<Type>(params)
    ;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx,dim)}
    };
    //config["phi"] = NSL::LinAlg::exp(
    //    I * NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim) 
    //);

    S.template getActionTerm<0>().sampleChiLike(config["phi"]);

    auto Sval = S(config);

    INFO(fmt::format("S = {}+i{}", NSL::real(Sval), NSL::imag(Sval)) );
    REQUIRE( almost_equal(NSL::RealTypeOf<Type>(0), NSL::imag(Sval), std::numeric_limits<Type>::digits10 - 1) ); 

    // This is how one computes the grad of the action
    NSL::Configuration<Type> gradS = S.grad(config);

    REQUIRE( 
        almost_equal(
            NSL::Tensor<NSL::RealTypeOf<Type>>(Nt,Nx,dim),S.force(config)["phi"].imag(),
            std::numeric_limits<Type>::digits10 - 1
        ).all() 
    ); 

    REQUIRE( ((gradS + S.force(config))["phi"] == 0 ).all() );

    NSL::RealTypeOf<Type> epsilon = 0.1;
    auto S_val = S(config);
    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            for(int d = 0; d < dim; ++d){
                NSL::Configuration<Type> configE(config,true);
                // let only a single link contribute
	            configE["phi"](t,i,d) += NSL::LinAlg::exp(I*epsilon);
                auto localS = S(configE);

                auto fin_diff = (localS-S_val)/epsilon;
                auto err = NSL::real(NSL::LinAlg::abs(fin_diff-gradS["phi"](t,i,d)));

                std::string repr = fmt::format("err= {}",err);
                INFO( repr );
                repr = fmt::format("t={},x={},d={}",t,i,d);
                INFO( repr );
                repr = fmt::format("Sε = {}+i{}",NSL::real(localS), NSL::imag(localS));
                INFO( repr );
                repr = fmt::format("ΔS = {}",NSL::to_string(fin_diff));
                INFO( repr );
                repr = fmt::format("∂S = {}",NSL::to_string(gradS["phi"](t,i,d)));
                INFO( repr );
                repr = fmt::format(" ε = {}",epsilon);
                INFO( repr );

                int fin_diff_order = getMatchingDigits(err);
                int eps_order = getMatchingDigits(epsilon);

                REQUIRE( fin_diff_order >= eps_order );

	            //  Note! this test will ALWAYS fail for complex<float>.
            }
        }
    }
}

template<typename Type>
void test_U1_Wilson_dMdPhi(){
    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type m    = 2.0;
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 8;
    //    Dimension of the System
    NSL::size_t dim = 2;
    // for larger lattices this fails as the finite difference fails

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1}; 

    NSL::Parameter params;
    params.addParameter<Type>("beta",beta);
    params.addParameter<Type>("bare mass",m);
    params.addParameter<NSL::size_t>("Nt",Nt);
    params.addParameter<NSL::size_t>("Nx",Nx);
    params.addParameter<NSL::size_t>("dim",dim);
    params.addParameter<NSL::Device>("device",NSL::CPU());

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    params.addParameter<NSL::Lattice::Square<Type>>("lattice",lattice);
    
    // Initialize the action
    NSL::FermionMatrix::U1::Wilson<Type> M(params);

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Tensor<Type> config = NSL::LinAlg::exp(
        I * NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim) 
    );

    M.populate(config);

    auto left = NSL::randn<Type>(Nt,Nx,dim);
    auto right = NSL::randn<Type>(Nt,Nx,dim);

    auto MM = M.M(right);
    auto dM = M.dMdPhi(left,right);

    NSL::RealTypeOf<Type> epsilon = 0.01;
    int eps_order = getMatchingDigits(epsilon);

    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            for(int d = 0; d < dim; ++d){
                NSL::Tensor<Type> configE(config,true);
                configE(t,i,d)*=NSL::LinAlg::exp(I*epsilon);

                M.populate(configE);

                auto MMe = M.M(right);

                auto dMe = (left * ((MMe - MM)/epsilon)).sum();

                auto err = NSL::LinAlg::abs(dMe - dM(t,i,d));

                INFO(fmt::format("err = {}", err));

                int order = getMatchingDigits(err);

                REQUIRE( order >= eps_order );


	            //  Note! this test will ALWAYS fail for complex<float>.
            }
        }
    }
}

template<typename Type>
void test_U1_Wilson_dMdaggerdPhi(){
    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type m    = 2.0;
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 8;
    //    Dimension of the System
    NSL::size_t dim = 2;
    // for larger lattices this fails as the finite difference fails

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1}; 

    NSL::Parameter params;
    params.addParameter<Type>("beta",beta);
    params.addParameter<Type>("bare mass",m);
    params.addParameter<NSL::size_t>("Nt",Nt);
    params.addParameter<NSL::size_t>("Nx",Nx);
    params.addParameter<NSL::size_t>("dim",dim);
    params.addParameter<NSL::Device>("device",NSL::CPU());

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    params.addParameter<NSL::Lattice::Square<Type>>("lattice",lattice);
    
    // Initialize the action
    NSL::FermionMatrix::U1::Wilson<Type> M(params);

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Tensor<Type> config = NSL::LinAlg::exp(
        I * NSL::randn<NSL::RealTypeOf<Type>>(Nt,Nx,dim) 
    );

    M.populate(config);

    auto left = NSL::randn<Type>(Nt,Nx,dim);
    auto right = NSL::randn<Type>(Nt,Nx,dim);

    auto MM = M.Mdagger(right);
    auto dM = M.dMdaggerdPhi(left,right);

    NSL::RealTypeOf<Type> epsilon = 0.01;
    int eps_order = getMatchingDigits(epsilon);

    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            for(int d = 0; d < dim; ++d){
                NSL::Tensor<Type> configE(config,true);
                configE(t,i,d)*=NSL::LinAlg::exp(I*epsilon);

                M.populate(configE);

                auto MMe = M.Mdagger(right);

                auto dMe = (left * ((MMe - MM)/epsilon)).sum();

                auto err = NSL::LinAlg::abs(dMe - dM(t,i,d));

                INFO(fmt::format("err = {}", err));

                int order = getMatchingDigits(err);

                REQUIRE( order >= eps_order );


	            //  Note! this test will ALWAYS fail for complex<float>.
            }
        }
    }
}

