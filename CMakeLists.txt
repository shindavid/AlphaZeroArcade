cmake_minimum_required(VERSION 3.17)

project(C4Experiment)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 ${TORCH_CXX_FLAGS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp")
include_directories("/home/dshin/anaconda3/envs/dev/include/eigen3")  # TODO: fixme, remove hard-coded path!

set(SOURCES cpp/main/C4Experiment.cpp)

add_executable(c4_experiment ${SOURCES})
target_link_libraries(c4_experiment "${TORCH_LIBRARIES}")
