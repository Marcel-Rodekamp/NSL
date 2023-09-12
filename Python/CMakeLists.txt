pybind11_add_module(PyNSLtest helloworld.cpp)
target_link_libraries(PyNSLtest PUBLIC NSL)

pybind11_add_module(PyNSL PyNSL.cpp)
target_link_libraries(PyNSL PUBLIC NSL)