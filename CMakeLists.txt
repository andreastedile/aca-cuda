cmake_minimum_required(VERSION 3.0)
project(aca_cuda CXX CUDA)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(argparse REQUIRED)
find_package(spdlog REQUIRED)
find_package(stb REQUIRED)

add_executable(app main.cu
        cpu_types.h
        gpu_types.h
        pixel.h
        rgb.h
        flatten.cpp
        flatten.h
        node.h
        utils.h
        construction.cu
        construction.cuh
        colorization.cpp
        colorization.h)

set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "60")

target_compile_definitions(app PUBLIC STB_IMAGE_IMPLEMENTATION)
target_compile_definitions(app PUBLIC STB_IMAGE_WRITE_IMPLEMENTATION)

target_link_libraries(app argparse::argparse)
target_link_libraries(app spdlog::spdlog)
target_link_libraries(app stb::stb)
