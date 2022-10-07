add_executable(test_typePromotion test_typePromotion.cpp ../test_main.cpp)
target_link_libraries(test_typePromotion NSL Catch2::Catch2)
add_test(NAME test_typePromotion COMMAND test_typePromotion)

