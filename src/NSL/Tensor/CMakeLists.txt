target_sources(NSL PUBLIC
        tensor.hpp
#        tensor.cpp
)

if(CUDA)
    message(" FOUND CUDA: ${CUDA}")
    target_sources(NSL PUBLIC
        tensor.cuh
#        tensor.cu
    )
endif()