cmake_minimum_required(VERSION 3.17)

project(C4Experiment)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS ${MY_TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem  REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 ${TORCH_CXX_FLAGS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

set(SOURCES cpp/main/GenerateTrainingData.cpp)

add_executable(c4_gen_training_data ${SOURCES})
add_subdirectory("cpp/util")
link_directories(${Boost_LIBRARY_DIRS})
target_link_libraries(c4_gen_training_data ${TORCH_LIBRARIES})
target_link_libraries(c4_gen_training_data ${Boost_LIBRARIES})

set_property(TARGET c4_gen_training_data PROPERTY CXX_STANDARD 20)

# https://stackoverflow.com/a/31390758/543913
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

print_all_variables()
