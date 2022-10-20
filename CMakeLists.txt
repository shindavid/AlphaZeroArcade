cmake_minimum_required(VERSION 3.17)

project(C4Experiment)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp")

set(SOURCES cpp/main/C4Experiment.cpp)

add_executable(c4_experiment ${SOURCES})
