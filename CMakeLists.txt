cmake_minimum_required(VERSION 3.18)
project(cuda_sgemm_optimization LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

include_directories(include)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Source files
file(GLOB SOURCES "src/*.cu" "src/kernels/*.cu")

# Executable
add_executable(sgemm_bench ${SOURCES})

# Link libraries
target_link_libraries(sgemm_bench PRIVATE CUDA::cudart)
