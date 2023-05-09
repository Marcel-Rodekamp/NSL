# relative include to the src/
target_include_directories(NSL PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(NSL spdlog::spdlog yaml-cpp::yaml-cpp)
