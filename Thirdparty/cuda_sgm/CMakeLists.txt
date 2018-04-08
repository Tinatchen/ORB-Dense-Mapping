cmake_minimum_required(VERSION 2.8)
project(cuda_sgm)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.5")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_definitions(-DCOMPILEDWITHC11)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_subdirectory(src)

add_subdirectory(Example)

