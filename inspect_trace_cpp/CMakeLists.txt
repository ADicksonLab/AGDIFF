cmake_minimum_required(VERSION 3.10)
project(hello-world)

# The first thing to do is to tell CMake to find the TorchScatter
# and TorchSparse libraries.
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(TorchCluster REQUIRED)

find_package(Python3 COMPONENTS Development)

# Include directories for header files
include_directories("/mnt/home/fathinia/.local/include")

add_executable(hello-world main.cpp)

# Link the TorchScatter and TorchSparse libraries to our executable
target_compile_features(hello-world PUBLIC cxx_range_for)
target_link_libraries(hello-world TorchScatter::TorchScatter)
target_link_libraries(hello-world TorchSparse::TorchSparse)
target_link_libraries(hello-world TorchCluster::TorchCluster) 
target_link_libraries(hello-world ${CUDA_cusparse_LIBRARY})
set_property(TARGET hello-world PROPERTY CXX_STANDARD 17)
