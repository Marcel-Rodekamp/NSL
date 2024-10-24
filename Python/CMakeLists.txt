# find_package(pybind11 REQUIRED)
message(TORCH_INSTALL_PREFIX="${TORCH_INSTALL_PREFIX}")

pybind11_add_module(PyNSLtest helloworld.cpp)
target_link_libraries(PyNSLtest PUBLIC NSL)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
pybind11_add_module(PyNSL PyNSL.cpp) #! Get teh cpp /home/physics/PhD/Projects/venvs/NSL/lib/python3.10/site-packages/torch/include/torch/csrc/utils/pybind.h
target_link_libraries(PyNSL PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} NSL)
target_compile_options(PyNSL PUBLIC ${TORCH_CXX_FLAGS})

file(GLOB PY_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
foreach(PY_FILE ${PY_FILES})
    get_filename_component(PY_FILE_NAME ${PY_FILE} NAME)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${PY_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${PY_FILE_NAME})
endforeach()
# file(COPY example_perylene.h5 DESTINATION ${CMAKE_CURRENT_BINARY_DIR})