
add_executable(example_TensorBasics example_tensor_basics.cpp)
target_link_libraries(example_TensorBasics NSL)

add_executable(example_GPU_Tensor example_GPU_Tensor.cpp)
target_link_libraries(example_GPU_Tensor NSL)

add_executable(example_Tensor_datalocality example_Tensor_datalocality.cpp)
target_link_libraries(example_Tensor_datalocality NSL)

add_executable(example_action example_action.cpp)
target_link_libraries(example_action NSL)

add_executable(example_MCMC example_MCMC.cpp)
target_link_libraries(example_MCMC NSL)

add_executable(example_RadialMCMC example_RadialMCMC.cpp)
target_link_libraries(example_RadialMCMC NSL)

#add_executable(example_GPU_Tensor example_GPU_Tensor.cpp)
#target_link_libraries(example_GPU_Tensor NSL)

add_executable(example_leapfrog example_leapfrog.cpp)
target_link_libraries(example_leapfrog NSL)

add_executable(example_rungeKutta4 example_rungeKutta4.cpp)
target_link_libraries(example_rungeKutta4 NSL)

add_executable(example_honeycomb example_honeycomb.cpp)
target_link_libraries(example_honeycomb NSL)

add_executable(example_CG example_CG.cpp)
target_link_libraries(example_CG NSL)

add_executable(example_BiCGStab example_BiCGStab.cpp)
target_link_libraries(example_BiCGStab NSL)

add_executable(example_IO example_IO.cpp)
target_link_libraries(example_IO NSL)

add_executable(example_lattice example_lattice.cpp)
target_link_libraries(example_lattice NSL)

add_executable(example_correlators example_correlators.cpp)
target_link_libraries(example_correlators NSL)

add_executable(example_cli example_cli.cpp)
target_link_libraries(example_cli NSL)

# Copy the example parameter file to the binary so that one can execute the 
# code without specifying a full path to the src folder.
file(COPY example_param.yml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
