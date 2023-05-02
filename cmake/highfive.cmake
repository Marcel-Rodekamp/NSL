FetchContent_Declare(
    HighFive 
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive
    GIT_TAG v2.7.1
)
FetchContent_MakeAvailable(HighFive)
target_link_libraries(NSL HighFive)





