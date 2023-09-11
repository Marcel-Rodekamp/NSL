pybind11_add_module(PyNSL helloworld.cpp)
target_link_libraries(PyNSL PUBLIC NSL)
