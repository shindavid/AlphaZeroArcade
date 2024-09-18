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

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds -O3 ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5 -ftemplate-backtrace-limit=50 ${EXTRA_DEFINITIONS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

# Suppress compiler warning about runtime search path:
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${MY_TORCH_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${MY_TORCH_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

include_directories("cpp/include")
include_directories("cpp")  # for inline/ subdirectory
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${MY_TINYEXPR_DIR})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_subdirectory("cpp/src")
add_subdirectory("cpp/src/games/blokus")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")

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
