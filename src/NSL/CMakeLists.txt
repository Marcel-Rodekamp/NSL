# relative include to the src/
target_include_directories(NSL PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(NSL PRIVATE
    assert.hpp
    complex.hpp
    concepts.hpp
    FermionMatrix.hpp
    Lattice.hpp
    LinAlg.hpp
    map.hpp
    Matrix.hpp
    NSL.hpp
    paramPack.hpp
    sliceObj.tpp
    Tensor.hpp
    types.hpp
)


# add the tensor class
add_subdirectory(Tensor/)
add_subdirectory(LinAlg/)
add_subdirectory(FermionMatrix/)
add_subdirectory(Lattice/)
