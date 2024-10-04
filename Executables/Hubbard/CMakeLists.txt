add_executable(flowForTrain_keepLast flowForTrain_keepLast.cpp)
target_link_libraries(flowForTrain_keepLast NSL)

add_executable(flowForTrain_keepProbable flowForTrain_keepProbable.cpp)
target_link_libraries(flowForTrain_keepProbable NSL)

add_executable(flowForTrain_stepwise flowForTrain_stepwise.cpp)
target_link_libraries(flowForTrain_stepwise NSL)

add_executable(HubbardHMC HMC.cpp)
target_link_libraries(HubbardHMC NSL)

add_executable(PseudoHMC PseudoHMC.cpp)
target_link_libraries(PseudoHMC NSL)

# Copy the example parameter file to the binary so that one can execute the 
# code without specifying a full path to the src folder.
file(COPY example_param.yml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})