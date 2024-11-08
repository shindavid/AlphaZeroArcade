cmake_minimum_required(VERSION 3.17)

project(alphazero)


if(CONDA_PREFIX)
    # Append to CMAKE_PREFIX_PATH and link_directories
    list(APPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
    link_directories(${CONDA_PREFIX}/lib)
endif()

# We enabled NDEBUG even for debug builds because a gcc bug leads to a spurious assert() deep
# in the eigen3 library
add_definitions(-DNDEBUG)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Torch REQUIRED PATHS ${MY_TORCH_DIR})
find_package(Boost COMPONENTS program_options filesystem json log log_setup REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_STANDARD  20)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -Wall -Wshadow -Wno-array-bounds ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5 -ftemplate-backtrace-limit=50 ${EXTRA_DEFINITIONS}")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp/include")
include_directories("cpp")  # for inline/ subdirectory
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

# Clear the targets file at the start of the configuration
file(WRITE ${CMAKE_BINARY_DIR}/targets.txt "")

# Function to append target metadata to the targets.txt file
function(append_target_metadata game_name)
    foreach(target ${ARGN})
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "EXECUTABLE")
            get_target_property(output_dir ${target} RUNTIME_OUTPUT_DIRECTORY)
            if(NOT output_dir)
                set(output_dir "${CMAKE_BINARY_DIR}/bin")
            endif()
            get_target_property(output_name ${target} OUTPUT_NAME)
            if(NOT output_name)
                set(output_name ${target})
            endif()
            file(APPEND ${CMAKE_BINARY_DIR}/targets.txt "${game_name} ${target} ${output_dir} ${output_name}\n")
        elseif(target_type STREQUAL "SHARED_LIBRARY")
            get_target_property(output_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
            if(NOT output_dir)
                set(output_dir "${CMAKE_BINARY_DIR}/lib")
            endif()
            get_target_property(output_name ${target} OUTPUT_NAME)
            if(NOT output_name)
                set(output_name ${target})
            endif()
            file(APPEND ${CMAKE_BINARY_DIR}/targets.txt "${game_name} ${target} ${output_dir} lib${output_name}.so\n")
        endif()
    endforeach()
endfunction()

add_subdirectory("cpp/src")
add_subdirectory("cpp/src/games/blokus")
#add_subdirectory("cpp/src/games/chess")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")
add_subdirectory("cpp/src/games/nim")
add_subdirectory("cpp/src/mcts")
add_subdirectory("cpp/src/generic_players")

# https://stackoverflow.com/a/31390758/543913
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# print_all_variables()
