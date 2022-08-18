cmake_minimum_required(VERSION 3.0)
project(aca_cuda CXX CUDA)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(argparse REQUIRED)
find_package(spdlog REQUIRED)
find_package(stb REQUIRED)

add_executable(app main.cu
        cpu_types.h
        gpu_types.h
        pixel.h
        rgb.h
        flatten.cpp flatten.cpp
        node.h )

set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(app PUBLIC STB_IMAGE_IMPLEMENTATION)
target_compile_definitions(app PUBLIC STB_IMAGE_WRITE_IMPLEMENTATION)

target_link_libraries(app argparse::argparse)
target_link_libraries(app spdlog::spdlog)
target_link_libraries(app stb::stb)
