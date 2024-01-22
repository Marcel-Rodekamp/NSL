FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11
        GIT_TAG        v2.3.2
)

FetchContent_MakeAvailable(cli11)
target_link_libraries(NSL CLI11::CLI11)
