cmake_minimum_required(VERSION 3.18)
project(Alloy CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ── Multi-architecture compilation ──
# sm86 = A6000, sm90 = H100
set(CMAKE_CUDA_ARCHITECTURES "86;90")

find_package(CUDAToolkit REQUIRED)

# ── HypeReca core (backbone) ──
add_subdirectory(core/hypereca EXCLUDE_FROM_ALL)

# ── Alloy tiered embedding library ──
add_library(alloy_kernels
    alloy/src/tiered_embedding.cu
)

target_include_directories(alloy_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/alloy/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/hypereca/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/hypereca/src
)

target_link_libraries(alloy_kernels
    CUDA::cudart
)

# Enable extended lambda and relaxed constexpr for device code
target_compile_options(alloy_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -v
    >
)
