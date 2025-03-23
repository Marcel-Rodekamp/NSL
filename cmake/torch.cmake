# check if torch can be found on the system default
#find_package(Torch REQUIRED)

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

option(USE_CPU "Use CPU instead of GPU" OFF)
option(USE_CUDA "Use GPU instead of CPU" OFF)
if (USE_CPU)
    message(STATUS "USE_CPU flag is set, disabling GPU optimizations")
    add_definitions(-DUSE_CPU)
elseif (USE_CUDA)
    message(STATUS "USE_CUDA flag is set, enabling GPU optimizations")
    add_definitions(-DUSE_CUDA)
else()
    # Check if Torch is built with CUDA support
    if (Torch_FOUND AND Torch_CUDA_VERSION)
        message(STATUS "Torch with CUDA support found, enabling GPU optimizations")
        add_definitions(-DUSE_CUDA)
    else()
        if (Torch_FOUND)
            message(STATUS "Torch without CUDA support found, disabling GPU optimizations")
        else()
            message(STATUS "Torch not found, disabling GPU optimizations")
        endif()
        add_definitions(-DUSE_CPU)
    endif()
endif()