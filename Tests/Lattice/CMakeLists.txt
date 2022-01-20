add_executable(test_ring test_ring.cpp ../test_main.cpp)
target_link_libraries(test_ring NSL Catch2::Catch2)
add_test(NAME test_ring COMMAND test_ring)

add_executable(test_square test_square.cpp ../test_main.cpp)
target_link_libraries(test_square NSL Catch2::Catch2)
add_test(NAME test_square COMMAND test_square)

add_executable(test_bipartite test_bipartite.cpp ../test_main.cpp)
target_link_libraries(test_bipartite NSL Catch2::Catch2)
add_test(NAME test_bipartite COMMAND test_bipartite)

add_executable(test_complete test_complete.cpp ../test_main.cpp)
target_link_libraries(test_complete NSL Catch2::Catch2)
add_test(NAME test_complete COMMAND test_complete)
