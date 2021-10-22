add_executable(test_ring test_ring.cpp)
target_link_libraries(test_ring NSL_TEST NSL)

add_executable(test_square test_square.cpp)
target_link_libraries(test_square NSL_TEST NSL)

add_executable(test_bipartite test_bipartite.cpp)
target_link_libraries(test_bipartite NSL_TEST NSL)

add_executable(test_complete test_complete.cpp)
target_link_libraries(test_complete NSL_TEST)
