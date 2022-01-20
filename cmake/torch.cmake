# check if torch can be found on the system default
find_package(Torch REQUIRED)

# if not found download a version and populate it
if(Torch_FOUND)
    message(STATUS "Found Installed Torch: ${TORCH_INCLUDE_DIRS}")
endif()

message(STATUS "Torch dir: ${Torch_DIR}")
message(STATUS "Adding TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "Adding TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(NSL
    ${TORCH_LIBRARIES}
)
