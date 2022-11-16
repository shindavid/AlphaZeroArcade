cmake_minimum_required(VERSION 3.17)

set(CMAKE_DEBUG_POSTFIX d)
project(project_name)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DNDEBUG)
endif()

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS ${MY_TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem  REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(c4_gen_training_data cpp/connect4/main/GenerateTrainingData.cpp)
add_executable(c4_play_vs_cpu cpp/connect4/main/PlayVsCpu.cpp)

add_subdirectory("cpp/util")
link_directories(${Boost_LIBRARY_DIRS})

target_link_libraries(c4_gen_training_data ${TORCH_LIBRARIES})
target_link_libraries(c4_gen_training_data ${Boost_LIBRARIES})
set_target_properties(c4_gen_training_data PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_property(TARGET c4_gen_training_data PROPERTY CXX_STANDARD 20)

target_link_libraries(c4_play_vs_cpu ${TORCH_LIBRARIES})
target_link_libraries(c4_play_vs_cpu ${Boost_LIBRARIES})
set_target_properties(c4_play_vs_cpu PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_property(TARGET c4_play_vs_cpu PROPERTY CXX_STANDARD 20)

# https://stackoverflow.com/a/31390758/543913
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

#print_all_variables()
