cmake_minimum_required(VERSION 3.17)

set(CMAKE_DEBUG_POSTFIX d)
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
find_package(Boost COMPONENTS program_options filesystem  REQUIRED)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-array-bounds -O3 ${TORCH_CXX_FLAGS} -fconcepts-diagnostics-depth=5 -ftemplate-backtrace-limit=50 ${EXTRA_DEFINITIONS}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)

include_directories("cpp/include")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${MY_EIGENRAND_DIR})
include_directories(${MY_TINYEXPR_DIR})

add_executable(c4 cpp/src/games/connect4/main/Connect4.cpp)
add_executable(othello cpp/src/games/othello/main/Othello.cpp)
add_executable(othello_unit_tests cpp/src/games/othello/main/UnitTests.cpp)
add_executable(tictactoe cpp/src/games/tictactoe/main/TicTacToe.cpp)
add_executable(util_unit_tests cpp/src/util/main/UnitTests.cpp)

set(executables
        c4
        othello
        othello_unit_tests
        tictactoe
        util_unit_tests
        )

add_subdirectory("cpp/src/mcts")
add_subdirectory("cpp/src/util")
add_subdirectory("cpp/src/games/connect4")
add_subdirectory("cpp/src/games/othello")
add_subdirectory("cpp/src/games/tictactoe")

set(common-libs mcts util)
set(other-libs connect4-lib othello-lib tictactoe-lib)

foreach(tgt ${executables} ${common-libs} ${other-libs})
    target_link_libraries(${tgt} ${TORCH_LIBRARIES})
    target_link_libraries(${tgt} ${Boost_LIBRARIES})
endforeach()

link_directories(${Boost_LIBRARY_DIRS})

foreach(tgt ${executables})
    if (${tgt} STREQUAL "c4")
        target_link_libraries(${tgt} connect4-lib)
    elseif (${tgt} STREQUAL "othello")
        target_link_libraries(${tgt} othello-lib)
    elseif (${tgt} STREQUAL "tictactoe")
        target_link_libraries(${tgt} tictactoe-lib)
    elseif (${tgt} STREQUAL "othello_unit_tests")
        target_link_libraries(${tgt} othello-lib)
    endif()
    foreach(lib ${common-libs})
        target_link_libraries(${tgt} ${lib})
    endforeach()
    set_target_properties(${tgt} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 20)
endforeach()

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
