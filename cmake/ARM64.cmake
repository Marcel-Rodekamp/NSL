if(APPLE)
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
    message(STATUS "Building for Architecture: ${ARCHITECTURE}" )
    if(${ARCHITECTURE} STREQUAL arm64)
        set(CMAKE_SYSTEM_NAME Darwin)
        set(CMAKE_HOST_SYSTEM_PROCESSOR arm64)
        set(CMAKE_SYSTEM_PROCESSOR arm64)
        set(triple arm64-apple-darwin20.6.0)
    
        set(CMAKE_C_COMPILER_TARGET ${triple})
        set(CMAKE_CXX_COMPILER_TARGET ${triple})
    endif()
endif()