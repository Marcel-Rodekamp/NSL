
add_executable(HMC HMC.cpp)
target_link_libraries(HMC NSL)

add_executable(HMC_pseudo HMC_pseudo.cpp)
target_link_libraries(HMC_pseudo NSL)

add_executable(Measure Measure.cpp)
target_link_libraries(Measure NSL)

add_executable(Measure_CC Measure_CC.cpp)
target_link_libraries(Measure_CC NSL)

#add_executable(StN StN.cpp)
#target_link_libraries(StN NSL)
add_executable(wilsonlogdet wilsonlogdet.cpp)
target_link_libraries(wilsonlogdet NSL)

# Copy the example parameter file to the binary so that one can execute the 
# code without specifying a full path to the src folder.
file(COPY example_param.yml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
