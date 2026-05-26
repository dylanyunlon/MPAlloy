cmake_minimum_required(VERSION 3.18)
project(MPAlloy LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86;90")

option(ALLOY_BUILD_TESTS "Build Alloy experiment binaries" ON)

# Core Alloy CUDA library
add_subdirectory(alloy)

if(ALLOY_BUILD_TESTS)
    # TODO: experiment runner binaries
endif()
