
add_executable(HMC HMC.cpp)
target_link_libraries(HMC NSL)

add_executable(Measure Measure.cpp)
target_link_libraries(Measure NSL)

add_executable(StN StN.cpp)
target_link_libraries(StN NSL)

# Copy the example parameter file to the binary so that one can execute the 
# code without specifying a full path to the src folder.
file(COPY example_param.yml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
