pybind11_add_module(PyNSLtest helloworld.cpp)
target_link_libraries(PyNSLtest PUBLIC NSL)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
pybind11_add_module(PyNSL PyNSL.cpp) #! Get teh cpp /home/physics/PhD/Projects/venvs/NSL/lib/python3.10/site-packages/torch/include/torch/csrc/utils/pybind.h
target_link_libraries(PyNSL PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} NSL)
target_compile_options(PyNSL PUBLIC ${TORCH_CXX_FLAGS})