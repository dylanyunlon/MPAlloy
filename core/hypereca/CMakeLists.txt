cmake_minimum_required(VERSION 3.8)
project(DrESS)

include_directories("include")

option("USE_CUDA" "Build DrESS with CUDA" ON)
option("USE_MPI" "Build DrESS with MPI" ON)
option("USE_UCWO" "Build DrESS with UCWO" ON)
option("BUILD_TEST" "Build DrESS tests" ON)
option("BUILD_BENCHMARK" "Build DrESS benchmarks" ON)
option("CMAKE_CUDA_ARCHITECTURES" "CUDA architecture code" "sm70,sm80")

file(GLOB_RECURSE SRCS
    src/*.cc
    src/*_impl.hh
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDRESS_USE_MPI")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDRESS_USE_MPI")
    find_package(MPI)
    include_directories(${MPI_INCLUDE_PATH})
endif()

if(USE_UCWO)
    if(!USE_MPI)
        message(FATAL_ERROR "UCWO has to be used with MPI")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDRESS_USE_UCWO")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDRESS_USE_UCWO")
    list(APPEND SRCS third_party/ucwo/src/ucwo.cc)
    include_directories(third_party/ucwo/include)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDRESS_USE_CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDRESS_USE_CUDA")
    file(GLOB_RECURSE CUDA_SRCS
        src/*.cu
        src/*_impl.cuh
    )
    set(CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND SRCS ${CUDA_SRCS})
endif()

if(BUILD_TEST)
    enable_testing()
    add_subdirectory("third_party/googletest")
    add_subdirectory("tests")
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()

add_library(dress SHARED ${SRCS})
set_property(TARGET dress PROPERTY CXX_STANDARD 17)

if(USE_MPI)
    target_link_libraries(dress ${MPI_C_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(dress nccl)
endif()

add_executable(reshard_test tests/reshard_test)
target_link_libraries(reshard_test dress ${MPI_C_LIBRARIES} nccl cudart)
