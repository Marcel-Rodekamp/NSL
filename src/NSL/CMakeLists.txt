# relative include to the src/
target_include_directories(NSL PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(NSL PUBLIC
        assert.hpp
        complex.hpp
        map.hpp
        )


# add the tensor class
add_subdirectory(Tensor/)
add_subdirectory(LinAlg/)
add_subdirectory(FermionMatrix/)
add_subdirectory(Lattice/)
