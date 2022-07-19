cmake_minimum_required(VERSION 3.0)
project(aca_cuda CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(argparse REQUIRED)
find_package(stb REQUIRED)

add_executable(app main.cpp)

set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(app argparse::argparse)
target_compile_definitions(app PRIVATE STB_IMAGE_IMPLEMENTATION)
target_link_libraries(app stb::stb)
