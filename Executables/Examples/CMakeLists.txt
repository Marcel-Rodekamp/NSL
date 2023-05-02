add_executable(example_action example_action.cpp)
target_link_libraries(example_action NSL)

add_executable(example_MCMC example_MCMC.cpp)
target_link_libraries(example_MCMC NSL)

#add_executable(example_GPU_Tensor example_GPU_Tensor.cpp)
#target_link_libraries(example_GPU_Tensor NSL)

add_executable(example_leapfrog example_leapfrog.cpp)
target_link_libraries(example_leapfrog NSL)

add_executable(example_CG example_CG.cpp)
target_link_libraries(example_CG NSL)

add_executable(example_IO example_IO.cpp)
target_link_libraries(example_IO NSL)
